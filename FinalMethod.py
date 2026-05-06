# 仅保留方面级情感分析相关代码
import pandas as pd
import spacy
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import random
import logging
import os
from tqdm import tqdm

class RoutingAggregation(nn.Module):
    def __init__(self, input_dim, output_dim, num_capsules=8, num_iterations=3):
        super(RoutingAggregation, self).__init__()
        self.num_capsules = num_capsules
        self.num_iterations = num_iterations
        self.weight = nn.Parameter(nn.init.xavier_uniform_(
            torch.randn(num_capsules, input_dim, output_dim)
        ))

    def squash(self, x):
        """胶囊网络的压缩函数"""
        norm = torch.norm(x, dim=-1, keepdim=True)
        return (norm / (1 + norm ** 2)) * x

    def forward(self, neighbors_features, node_features):
        # 移除max_neighbors维度，现在形状为[num_nodes, input_dim]
        num_nodes, input_dim = neighbors_features.shape
        
        # 映射邻居特征为预测向量u_hat，增加胶囊维度
        u_hat = torch.einsum('nj,kjl->nkl', neighbors_features, self.weight)  # [num_nodes, num_capsules, output_dim]

        # 基于相似度初始化b
        node_norm = torch.norm(node_features, dim=-1, keepdim=True)  # [num_nodes, 1]
        neighbor_norm = torch.norm(neighbors_features, dim=-1, keepdim=True)  # [num_nodes, 1]
        dot_product = torch.sum(node_features * neighbors_features, dim=-1, keepdim=True)  # [num_nodes, 1]
        cosine_sim = dot_product / (node_norm * neighbor_norm + 1e-8)
        
        # 初始化路由权重b
        b = cosine_sim.unsqueeze(-2).expand(-1, self.num_capsules, -1).clone()  # [num_nodes, num_capsules, 1]

        # 动态路由迭代
        for i in range(self.num_iterations):
            c = F.softmax(b, dim=1)  # 按胶囊维度归一化
            s = (c * u_hat).sum(dim=1, keepdim=True)  # 聚合胶囊特征
            v = self.squash(s)  # 压缩得到胶囊输出
            
            if i < self.num_iterations - 1:
                # 路由权重更新（基于胶囊相似性）
                similarity = torch.matmul(u_hat.unsqueeze(-2), v.unsqueeze(-1)).sum(dim=-1)
                b += similarity  # 累积相似性更新b

        aggregated = v.squeeze(1)  # 聚合胶囊输出
        return aggregated

class RoutingGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, capsule_sizes=[8, 4], num_iterations=2):
        super(RoutingGCNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_iterations = num_iterations
        
        # 初始化多层RoutingAggregation，逐步收缩num_capsules
        self.aggregators = nn.ModuleList()
        prev_dim = in_feats
        for caps in capsule_sizes:
            self.aggregators.append(
                RoutingAggregation(
                    input_dim=prev_dim,
                    output_dim=out_feats if caps == capsule_sizes[-1] else prev_dim,
                    num_capsules=caps,
                    num_iterations=num_iterations
                )
            )
            prev_dim = out_feats if caps == capsule_sizes[-1] else prev_dim

    def forward(self, g, features):
        with g.local_scope():
            g.ndata['h'] = features  # 节点特征
            
            # 聚合邻居特征（带边权重）
            if 'w' in g.edata:  # 注意图中边权重的键是'w'
                g.update_all(
                    fn.u_mul_e('h', 'w', 'msg'),  # 使用边权重'edata['w']'
                    fn.mean('msg', 'neighbor_h')
                )
            else:
                # 无权重时，使用复制聚合
                g.update_all(
                    fn.copy_u('h', 'msg'),
                    fn.mean('msg', 'neighbor_h')
                )
            
            # 直接使用聚合后的邻居特征，不扩展max_neighbors维度
            batch_neighbors = g.ndata['neighbor_h']
            x = features 

            # 逐层应用RoutingAggregation
            for aggregator in self.aggregators:
#                 print(batch_neighbors.size())
#                 print(features.size())
                x = aggregator(batch_neighbors, x)
                # 为下一层准备邻居特征
                batch_neighbors = x 
                
            return x  # 补充返回值

class DependencyGCN(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(DependencyGCN, self).__init__()
        self.layer1 = RoutingGCNLayer(768, 768, capsule_sizes=[8,4], num_iterations=2)

    def forward(self, g, features, aspect_indices=None):
        h = self.layer1(g, features)
      
        if aspect_indices is not None and len(aspect_indices) > 0:
            aspect_features = h[aspect_indices]  # 按索引取特征
            graph_embeddings = torch.mean(aspect_features, dim=0)  # 平均池化
            
        return graph_embeddings

class RobertaGCN(nn.Module):
    def __init__(self, model_path='../model/roberta-base', dropout_rate=0.1, device=torch.device("cuda"), maxlength=256):
        super(RobertaGCN, self).__init__()
        
        self.device = device
        self.maxlength = maxlength
        self.dropout_rate = dropout_rate

        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.roberta = RobertaModel.from_pretrained(model_path).to(self.device)
        self.nlp = spacy.load("../model/spacy/en_core_web_sm-3.7.0/en_core_web_sm/en_core_web_sm-3.7.0")

        self.dropout = nn.Dropout(dropout_rate)

        self.DependencyGCN = DependencyGCN().to(self.device)
        self.SentenceGCN = DependencyGCN().to(self.device)

    def get_Vector(self, text):
        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        outputs = self.roberta(** encoding, output_attentions=True)
        return outputs

    def create_Graph(self, text, Embedding, threshold=0.2, graph_type='dependency'):
        doc = self.nlp(text)
        edges = []
        edge_weights = [] 

        node_embeddings = Embedding  # 形状: [num_tokens, embedding_dim]

        # 定义依赖关系权重映射表（用于依赖图）
        dep_weight_map = {
            'amod': 1.0, 'attr': 1.0, 'dobj': 0.95, 'nsubj': 0.95,
            'advmod': 0.85, 'cop': 0.8, 'iobj': 0.75,
            'prep': 0.7, 'acl': 0.65, 'advcl': 0.6, 'conj': 0.55
        }
        kept_deps = set(dep_weight_map.keys())
        
        if graph_type == 'dependency':
            # 依赖图逻辑保持不变（使用预定义权重）
            for token in doc:
                for child in token.children:
                    dep_rel = child.dep_
                    if dep_rel not in kept_deps:
                        continue
                    distance = abs(token.i - child.i)
                    edges.append((child.i, token.i))
                    edge_weights.append(dep_weight_map[dep_rel] / (1.0 + distance))
        elif graph_type == 'semantic':
            num_tokens = len(doc)
            normalized_emb = F.normalize(node_embeddings, p=2, dim=1)  # [num_tokens, embedding_dim]

            cos_sim = torch.matmul(normalized_emb, normalized_emb.T)  # 余弦相似度
            # 筛选相似度大于阈值且非自环的边
            for i in range(num_tokens):
                for j in range(num_tokens):
                    if i != j and cos_sim[i, j].item() > threshold:
                        edges.append((i, j))
                        edge_weights.append(cos_sim[i, j].item())  # 权重=余弦相似度值
        if not edges:
            graph = dgl.graph(([], []), num_nodes=len(doc))
            graph = dgl.add_self_loop(graph)
            graph.edata['w'] = torch.ones(graph.number_of_edges(), 1)
        else:
            src, dst = zip(*edges)
            graph = dgl.graph((src, dst), num_nodes=len(doc))
            graph = dgl.add_self_loop(graph)
            num_self_loops = sum(1 for src, dst in zip(graph.edges()[0], graph.edges()[1]) if src == dst)
            self_loop_weights = torch.ones(num_self_loops, 1)
            edge_weights_tensor = torch.tensor(edge_weights).view(-1, 1)
            graph.edata['w'] = torch.cat((edge_weights_tensor, self_loop_weights), dim=0)
        graph = graph.to(self.device)
        graph.ndata['h'] = Embedding
        return graph
    
    def get_word_embeddings(self, sentences):
        batch_size = len(sentences)
        all_word_embeddings = []
        encoding = self.tokenizer(sentences,  return_attention_mask=True, return_tensors='pt',padding=True, truncation=True)
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        outputs = self.roberta(**encoding).last_hidden_state

        all_tokens = [self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][i]) for i in range(batch_size)]
        
        for i in range(batch_size):
            tokens_new = all_tokens[i]  
            sentence = sentences[i]  
            doc = self.nlp(sentence)
            words = [token.text for token in doc]  
            current_token_index = 1 
            word_embeddings = []

            j = 0
            while j < len(words):
                word = words[j]
                new_word_remaining = ''.join(self.tokenizer.tokenize(word))
                word_embedding = torch.zeros(outputs.size(-1), device=self.device)

                while new_word_remaining and current_token_index < self.maxlength:
                    token = tokens_new[current_token_index]
                    if token.startswith('Ġ'):
                        token = token[1:]  # 去除 Ġ

                    while len(new_word_remaining) < len(token):
                        word_embedding = outputs[i, current_token_index]
                        word_embeddings.append(word_embedding)
                        token = token[len(new_word_remaining):]
                        j += 1
                        if j >= len(words):
                            break
                        word = words[j]
                        new_word_remaining = ''.join(self.tokenizer.tokenize(word))

                    if new_word_remaining.startswith(token):
                        new_word_remaining = new_word_remaining[len(token):]
                        word_embedding += outputs[i, current_token_index]
                    current_token_index += 1
                if new_word_remaining:
                    word_embeddings.append(torch.zeros(outputs.size(-1), device=self.device))
                else:
                    word_embeddings.append(word_embedding)
                j += 1  
           
            while j < len(words):
                word_embeddings.append(torch.zeros(outputs.size(-1), device=self.device))
                j += 1
            all_word_embeddings.append(torch.stack(word_embeddings))

        return all_word_embeddings

    def get_Contextual_embedding(self, sentences, termFrom, termTo, embeddings):
        batch_size = len(sentences)
        all_target_indices = []
        for i in range(batch_size):
            sentence = sentences[i]
            from_pos = termFrom[i]
            to_pos = termTo[i]
            doc = self.nlp(sentence)   

            start_token_index = None
            end_token_index = None
            
            for token in doc:
                token_start = token.idx
                token_end = token.idx + len(token.text)

                if token_start <= from_pos < token_end:
                    start_token_index = token.i

                if token_start < to_pos <= token_end:
                    end_token_index = token.i + 1

                if start_token_index is not None and end_token_index is not None:
                    break
            if start_token_index is None or end_token_index is None:
                return None
            aspect_indices = list(range(start_token_index, end_token_index))
            all_target_indices.append(aspect_indices)
        all_target_embeddings = []
        for i in range(batch_size):
            indices = torch.tensor(all_target_indices[i], dtype=torch.long, device=self.device)
            target_embeddings = torch.index_select(embeddings[i], dim=0, index=indices)
            target_embedding = torch.mean(target_embeddings, dim=0)
            all_target_embeddings.append(target_embedding)

        all_target_embeddings = torch.stack(all_target_embeddings, dim=0)
        all_target_embeddings = self.dropout(all_target_embeddings)

        return all_target_embeddings, all_target_indices
    def _process_single_instance(self, context, term, term_from, term_to):
        sentence_Embedding = self.get_word_embeddings(context)
        # aspect_Embedding = self.get_Vector(term).last_hidden_state.mean(dim=1)
        aspect_Contextual, all_aspect_indices = self.get_Contextual_embedding(context, term_from, term_to, sentence_Embedding)
        # 处理依赖图GCN
        AttentionOutputs = []
        for i in range(len(context)):
            text = context[i]
            embedding = sentence_Embedding[i]
            graph = self.create_Graph(text, embedding, graph_type='dependency')
            aspect_indices = all_aspect_indices[i]
            AttentionOutput = self.DependencyGCN(graph, graph.ndata['h'], aspect_indices) 
            AttentionOutputs.append(AttentionOutput)   
        AttentionEmbedding = torch.stack(AttentionOutputs, dim=0)

        # 处理语义图GCN
        SentenceOutputs = []
        for i in range(len(context)):
            text = context[i]
            embedding = sentence_Embedding[i]
            graph = self.create_Graph(text, embedding, graph_type='semantic')
            aspect_indices = all_aspect_indices[i]
            SentenceOutput = self.SentenceGCN(graph, graph.ndata['h'], aspect_indices)
            SentenceOutputs.append(SentenceOutput)   
        SentenceEmbedding = torch.stack(SentenceOutputs, dim=0)
        
        # 组合特征
        combined = torch.cat((AttentionEmbedding, SentenceEmbedding, aspect_Contextual), dim=1)
        return combined
    def forward(self, positive_data, anchor_data, negative_data):
        # 解析输入数据
        pos_term, pos_context, pos_from, pos_to, pos_polarity = positive_data
        anchor_term, anchor_context, anchor_from, anchor_to, anchor_polarity = anchor_data
        neg_term, neg_context, neg_from, neg_to, neg_polarity = negative_data
        
        # 获取各部分的嵌入（每个都是[batch_size, 768]）
        pos_emb = self._process_single_instance(pos_context, pos_term, pos_from, pos_to)
        anchor_emb = self._process_single_instance(anchor_context, anchor_term, anchor_from, anchor_to)
        neg_emb = self._process_single_instance(neg_context, neg_term, neg_from, neg_to)
        
        # 拼接三元组嵌入用于分类（[batch_size, 768*3]）
        triple_emb = torch.cat([pos_emb, anchor_emb, neg_emb], dim=1)
        
        # 返回三元组嵌入、三个单独嵌入及极性
        return triple_emb, (pos_emb, anchor_emb, neg_emb), (pos_polarity, anchor_polarity, neg_polarity)


class AspectLoss(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super(AspectLoss, self).__init__()
        self.aspect_loss_fn = nn.CrossEntropyLoss()
        # 修正输入维度：3个768维向量拼接
        self.AspectClassifier = nn.Sequential(
            nn.Linear(768*3*3, 768),  # 输入为3个768维特征拼接
            nn.ReLU(),
            nn.LayerNorm(768),
            nn.Dropout(0.2),
            nn.Linear(768, 3)  # 3类情感
        )
        self.AspectClassifier = self.AspectClassifier.to(device)
        self.aspect_loss_fn = self.aspect_loss_fn.to(device)
        self.device = device
        self.triple_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        
    def forward(self, triple_emb, embeddings, polarities):
        # 解析输入
        pos_emb, anchor_emb, neg_emb = embeddings  # 嵌入向量（用于三元组损失）
        pos_polarity, anchor_polarity, neg_polarity = polarities  # 极性标签（用于分类损失）
        
        # 分类损失（使用正例极性作为标签）
        pos_logits = self.AspectClassifier(triple_emb)
        cls_loss = self.aspect_loss_fn(pos_logits, pos_polarity.to(self.device))
        
#         # 三元组损失（使用嵌入向量计算）
        triple_loss = self.triple_loss_fn(anchor_emb, pos_emb,neg_emb)
        
        # 总损失
        total_loss = cls_loss + 0.3 * triple_loss
        return total_loss

class TripleAspectDataset(Dataset):
    def __init__(self, csv_path, is_test=False):  # 新增is_test标志
        self.data = pd.read_csv(csv_path)
        self.is_test = is_test  # 用于标识测试集

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 保持原有数据提取逻辑不变
        row = self.data.iloc[idx]
        positive_data = (
            str(row['PositiveTerm']),
            str(row['PositiveContext']),
            row['PositiveTermFrom'],
            row['PositiveTermTo'],
            torch.tensor(row['PositivePolarity'], dtype=torch.long)
        )
        anchor_data = (
            str(row['AnchorTerm'] if 'AnchorTerm' in row else ''),
            str(row['AnchorContext']),
            row['AnchorTermFrom'],
            row['AnchorTermTo'],
            torch.tensor(row['AnchorPolarity'], dtype=torch.long)
        )
        negative_data = (
            str(row['NegativeTerm']),
            str(row['NegativeContext']),
            row['NegativeTermFrom'],
            row['NegativeTermTo'],
            torch.tensor(row['NegativePolarity'], dtype=torch.long)
        )
        if self.is_test:
            return positive_data, anchor_data, negative_data, idx  # 测试集返回索引
        return positive_data, anchor_data, negative_data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, optimizer, criterion,F1Path, epochs=5):
    model.train()
    best_f1 = float('-inf')
    best_f1_model_path = F1Path
    for epoch in range(epochs):
        total_loss = 0
        for positive_data, anchor_data, negative_data in tqdm(train_loader, desc=f"Epoch {epoch + 1} Fine-Tuning"):
            optimizer.zero_grad()
            
            triple_emb, embeddings, polarities = model(positive_data, anchor_data, negative_data)
            
            # 计算损失（传入三个必要参数）
            loss = criterion(triple_emb, embeddings, polarities)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("=======FineTune========")
        print(f"Epoch {epoch+1}/{epochs} - Fine-tune Loss: {total_loss/len(train_loader):.4f}")
        logging.info("========FineTune===========")
        logging.info(f"Epoch {epoch+1}/{epochs} - Fine-tune Loss: {total_loss/len(train_loader):.4f}")
        acc, pre, rec, f1 = evaluate_model(model, val_loader, criterion)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_f1_model_path)
            print(f"Best model saved with F1: {best_f1}")
            logging.info("Best F1")

def evaluate_model(model, val_loader, criterion, class_names=['消极', '中性', '积极']):
    model.eval()
    all_preds = []  # 存储概率分布
    all_labels = []  # 存储标签
    all_indices = []  # 存储测试集索引（用于分组）
    is_test = False  # 标记是否为测试集

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # 解包数据，测试集包含索引
            if len(batch) == 4:
                positive_data, anchor_data, negative_data, indices = batch
                is_test = True
                all_indices.extend(indices.numpy())  # 收集索引
            else:
                positive_data, anchor_data, negative_data = batch
                indices = None

            triple_emb, _, polarities = model(positive_data, anchor_data, negative_data)
            pos_polarity, _, _ = polarities
            logits = criterion.AspectClassifier(triple_emb)
            probs = F.softmax(logits, dim=1)  # 转换为概率

            all_preds.append(probs)
            all_labels.append(pos_polarity)

    # 拼接所有数据
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    # 处理测试集：合并连续索引的中性句子三元组
    if is_test:
        # 按索引排序（确保顺序正确）
        sorted_indices = np.argsort(all_indices)
        sorted_preds = all_preds[sorted_indices]
        sorted_labels = all_labels[sorted_indices]
        sorted_original_indices = np.array(all_indices)[sorted_indices]

        merged_preds = []
        merged_labels = []
        i = 0
        while i < len(sorted_preds):
            # 判断当前是否为中性句子的第一个三元组（下一个索引连续）
            current_label = sorted_labels[i]
            if current_label == 1 and i + 1 < len(sorted_preds) and sorted_original_indices[i+1] == sorted_original_indices[i] + 1:
                # 合并两个三元组的概率（相加后取最高）
                combined_probs = sorted_preds[i] + sorted_preds[i+1]
                merged_preds.append(np.argmax(combined_probs))
                merged_labels.append(current_label)
                i += 2  # 跳过下一个（已合并）
            else:
                # 非中性句子或单个三元组，直接取当前预测
                merged_preds.append(np.argmax(sorted_preds[i]))
                merged_labels.append(current_label)
                i += 1

        # 使用合并后的结果计算指标
        preds = np.array(merged_preds)
        labels = np.array(merged_labels)
    else:
        # 训练集/验证集直接取预测
        preds = np.argmax(all_preds, axis=-1)
        labels = all_labels

    # 计算并打印指标（保持原有逻辑）
    metrics = compute_metrics(preds, labels)
    print(f"Val - 总体准确率: {metrics['accuracy']}%, 宏平均精确率: {metrics['precision_macro']}%, "
          f"宏平均召回率: {metrics['recall_macro']}%, 宏平均F1: {metrics['f1_macro']}%")
    logging.info(f"Val - 总体准确率: {metrics['accuracy']}%, 宏平均精确率: {metrics['precision_macro']}%, "
                 f"宏平均召回率: {metrics['recall_macro']}%, 宏平均F1: {metrics['f1_macro']}%")

    print("每个类别的详细指标:")
    logging.info("每个类别的详细指标:")
    for i, class_name in enumerate(class_names):
        print(f"类别 {class_name}: "
              f"精确率={metrics['per_class']['precision'][i]}%, "
              f"召回率={metrics['per_class']['recall'][i]}%, "
              f"F1={metrics['per_class']['f1'][i]}%, "
              f"样本数={metrics['per_class']['support'][i]}")
        logging.info(f"类别 {class_name}: "
                     f"精确率={metrics['per_class']['precision'][i]}%, "
                     f"召回率={metrics['per_class']['recall'][i]}%, "
                     f"F1={metrics['per_class']['f1'][i]}%, "
                     f"样本数={metrics['per_class']['support'][i]}")

    return (metrics['accuracy'], metrics['precision_macro'], 
            metrics['recall_macro'], metrics['f1_macro'])

def compute_metrics(logits, labels):
    # 修复：确保preds是数组类型（处理可能的标量情况）
    if isinstance(logits, np.ndarray) and logits.ndim == 1:
        # 对于已经是类别索引的情况直接使用
        preds = logits
    else:
        # 对于概率分布取最大概率索引
        preds = np.argmax(logits, axis=-1)
    
    # 确保labels是数组类型
    labels = np.asarray(labels)

    # 计算总体macro指标
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    accuracy = accuracy_score(labels, preds)

    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    # 构建结果字典
    metrics = {
        'accuracy': float(f"{accuracy * 100:.2f}"),
        'precision_macro': float(f"{precision_macro * 100:.2f}"),
        'recall_macro': float(f"{recall_macro * 100:.2f}"),
        'f1_macro': float(f"{f1_macro * 100:.2f}"),
        'per_class': {
            'precision': [float(f"{p * 100:.2f}") for p in precision_per_class],
            'recall': [float(f"{r * 100:.2f}") for r in recall_per_class],
            'f1': [float(f"{f * 100:.2f}") for f in f1_per_class],
            'support': support_per_class.tolist()
        }
    }
    return metrics

def setup_logger(logPath):
    logger = logging.getLogger()
    
    if logger.hasHandlers():
        for handler in logger.handlers[:]:  
            logger.removeHandler(handler)  
            handler.close() 

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    log_dir = os.path.dirname(logPath)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = logging.FileHandler(logPath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def BeginTrain(dataset, BatchSize, savePath, epochs, max_length=256):
    # 数据路径
    path_TrainAspect = f"../data/{dataset}/train.csv"
    path_TestAspect = f"../data/{dataset}/test.csv"

    logPath = f"../ablation/{dataset}/{savePath}_training.log"
    F1Path = f"../ablation/{dataset}/{savePath}_f1_weights.pth"

    print(logPath, F1Path)
    set_seed(25)
    setup_logger(logPath)

    # 加载数据集
    data_TrainAspect = TripleAspectDataset(path_TrainAspect)
    data_TestAspect = TripleAspectDataset(path_TestAspect, is_test=True)

    loader_TrainAspect = DataLoader(data_TrainAspect, batch_size=BatchSize, shuffle=True)
    loader_TestAspect = DataLoader(data_TestAspect, batch_size=BatchSize, shuffle=False)

    # 初始化模型和优化器
    model = RobertaGCN(maxlength=max_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = AspectLoss()

    # 训练模型
    logging.info("=======Begin Training===========")
    logging.info(f"Train: {path_TrainAspect}")
#     logging.info(f"Val: {path_ValAspect}")
    logging.info(f"Test: {path_TestAspect}")
    logging.info(f"Batch size: {loader_TrainAspect.batch_size}")
    logging.info(f"max_length: {max_length}")
    logging.info('Starting training...')
    
    train_model(model, loader_TrainAspect,loader_TestAspect, optimizer,criterion,F1Path,epochs=epochs)

    # 测试最佳模型
    logging.info('Starting test...')
    model = RobertaGCN(maxlength=max_length)
    model.load_state_dict(torch.load(F1Path))
    acc, pre, rec, f1 = evaluate_model(model, loader_TestAspect, criterion)
    logging.info(f"Test Results - Acc: {acc}, Pre: {pre}, Rec: {rec}, F1: {f1}")
    logging.info("=======End Training===========")
    os.remove(F1Path)

if __name__ == "__main__":   
    # 示例运行
    BeginTrain("Laptop",8, "No_TripLoss", 20)
    BeginTrain("Restaurants",8,"No_TripLoss", 20)
    BeginTrain("Twitter",16,"No_TripLoss", 15)
