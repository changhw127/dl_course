import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from matplotlib.gridspec import GridSpec
from deal_data import PARALLEL_DATA, MyData, tokenize_zh, tokenize_en
from my_transformer import encode_sentence, Transformer
from utlis.globals import load_model, get_device

import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
matplotlib.rcParams['font.family'] = 'Heiti TC'  # 可以替换为其他字体


def visualize_all_heads(model, src_input, src_tokens, tgt_input, tgt_tokens, tgt_mask, layer_idx=0):
    """
    可视化指定层所有注意力头的权重在一张图上

    参数:
        model: Transformer模型
        src_sentence: 源语言句子(英文)
        tgt_sentence: 目标语言句子(中文)
        data_obj: 数据对象(包含词汇表等信息)
        layer_idx: 要可视化的层索引(从0开始)
    """
    # device = next(model.parameters()).device
    #
    # # 编码输入句子
    # src_tokens = tokenize_en(src_sentence)
    # src_encoded = encode_sentence(src_tokens, data_obj.vocab_en, data_obj.word2idx_en)
    # src_input = src_encoded.unsqueeze(0).to(device)
    #
    # tgt_tokens = tokenize_zh(tgt_sentence)  # 假设tgt_sentence已经是中文分词后的字符串
    # tgt_encoded = encode_sentence(tgt_tokens, data_obj.vocab_zh, data_obj.word2idx_zh)
    # tgt_input = tgt_encoded[:-1].unsqueeze(0).to(device)
    #
    # # 生成目标掩码
    # tgt_seq_len = tgt_input.size(1)
    # tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).to(device)

    # 前向传播以获取注意力权重
    with torch.no_grad():
        model(src_input, tgt_input, src_mask=None, tgt_mask=tgt_mask)

    # 获取注意力权重
    encoder_attn = model.encoder.layers[layer_idx].self_attn.attention_weights[0]  # (num_heads, seq_len, seq_len)
    decoder_self_attn = model.decoder.layers[layer_idx].self_attn.attention_weights[0]
    decoder_cross_attn = model.decoder.layers[layer_idx].cross_attn.attention_weights[0]

    num_heads = encoder_attn.shape[0]

    # 创建大图
    plt.figure(figsize=(20, 6 * num_heads))
    plt.suptitle(f'Attention Weights Visualization - Layer {layer_idx + 1}', y=1.02, fontsize=16)

    # 使用GridSpec布局
    gs = GridSpec(num_heads, 3, width_ratios=[1, 1, 1], height_ratios=[1] * num_heads)

    for head in range(num_heads):
        # 编码器自注意力
        ax1 = plt.subplot(gs[head, 0])
        plot_single_head(encoder_attn[head], src_tokens, src_tokens,
                         ax=ax1, title=f'Encoder Head {head + 1}')

        # 解码器自注意力
        ax2 = plt.subplot(gs[head, 1])
        plot_single_head(decoder_self_attn[head], tgt_tokens[:-1], tgt_tokens[:-1],
                         ax=ax2, title=f'Decoder Self Head {head + 1}')

        # 解码器-编码器交叉注意力
        ax3 = plt.subplot(gs[head, 2])
        plot_single_head(decoder_cross_attn[head], src_tokens, tgt_tokens[:-1],
                         ax=ax3, title=f'Decoder Cross Head {head + 1}')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'results/my_transformer_details_weight_layer{layer_idx}.png', dpi=200)

def plot_single_head(weights, x_labels, y_labels, ax=None, title=""):
    """
    绘制单个注意力头的热力图

    参数:
        weights: 注意力权重矩阵 (seq_len_q, seq_len_k)
        x_labels: x轴标签(键)
        y_labels: y轴标签(查询)
        ax: matplotlib轴对象
        title: 子图标题
    """
    if ax is None:
        ax = plt.gca()

    sns.heatmap(weights, cmap='viridis',
                xticklabels=x_labels,
                yticklabels=y_labels,
                linewidths=0.1, linecolor='gray',
                cbar=False, ax=ax)

    ax.set_title(title, pad=10)
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_yticklabels(y_labels, rotation=0)


if __name__ == '__main__':
    # 加载训练好的模型
    device = get_device()
    data_obj = MyData(PARALLEL_DATA)
    all = [(src_sentence, tgt_sentence) for src_sentence, tgt_sentence in zip(data_obj.encoded_en, data_obj.encoded_zh)]
    ix = 2
    src_sentence, tgt_sentence = all[ix]
    src_sentence_ = [data_obj.idx2word_en[x.item()] for x in src_sentence]
    tgt_sentence_ = [data_obj.idx2word_zh[x.item()] for x in tgt_sentence]
    print(data_obj.parallel_data[ix][0])
    print(data_obj.parallel_data[ix][1])
    max_len = 20

    src_input = src_sentence.unsqueeze(0).to(device)  # tensor([[ 1, 11, 12, 13,  6,  2,  0]], device='mps:0')
    tgt_input = tgt_sentence[:-1].unsqueeze(0).to(device)  # tensor([[1, 7, 8, 2, 0, 0]], device='mps:0')
    tgt_output = tgt_sentence[1:].unsqueeze(0).to(device)  # tensor([[7, 8, 2, 0, 0, 0]], device='mps:0')

    tgt_seq_len = tgt_input.size(1)  # 6
    tgt_mask = Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt_input.device).to(dtype=torch.float32)

    model_args = {'d_model': 64, 'num_heads': 4, 'num_encoder_layers': 2, 'num_decoder_layers': 2,
                  'd_ff': 128, 'dropout': 0.1, 'src_vocab_size': 27, 'tgt_vocab_size': 29}

    model = load_model(Transformer, 'results/my_transformer_details.pth', 'mps', **model_args)

    # 可视化第0层所有注意力头
    # visualize_all_heads(model, src_input, src_sentence, tgt_input, tgt_sentence, tgt_mask, layer_idx=0)

    # 可视化所有层
    for layer in range(model_args['num_encoder_layers']):
        visualize_all_heads(model, src_input, src_sentence_, tgt_input, tgt_sentence_, tgt_mask, layer_idx=layer)