import argparse

def parse():
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec source')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='input batch size for training')
    parser.add_argument('--train_neg_num', type=int, default=1, metavar='N', help='the number of training negative samples')
    parser.add_argument('--test_neg_num', type=int, default=99, metavar='N',
                        help='the number of test negative samples')
    parser.add_argument('--embed_id_dim', type=int, default=128, metavar='N', help='ID embedding size')
    parser.add_argument('--text_embed_dim', type=int, default=384, metavar='N',
                        help='the embedding dim of text feature')
    parser.add_argument('--visual_embed_dim', type=int, default=4096, metavar='N',
                        help='the embedding dim of visual feature')
    parser.add_argument('--review_embed_dim', type=int, default=384, metavar='N',
                        help='the embedding dim of review feature')
    parser.add_argument('--domain_disen_dim', type=int, default=256, metavar='N',
                        help='the embedding dim of disentanglement domain feature')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train')
    parser.add_argument('--top_k', type=int, default=10, metavar='N', help='test top k')
    parser.add_argument('--field', type=str, default='amazon', metavar='N', help='datasets field')
    parser.add_argument('--datasets', type=str, default='elec_cloth', metavar='N', help='datasets')
    parser.add_argument('--alpha', type=float, default=0.005,
                        help='the weight of contrastive loss')
    parser.add_argument('--ldp_noise', type=float, default=0.01,
                        help='the noise control of LDP')
    parser.add_argument('--ssl_temp', type=float, default=0.1,
                        help='the temperature coefficient')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='the dropout rate')
    parser.add_argument('--n_layers', type=int, default=3, metavar='N', help='the number of GNN layers')
    parser.add_argument('--agg_way', type=str, default='concat', metavar='N',
                        help='aggregation way of disentangled features')


    args = parser.parse_args()
    return args