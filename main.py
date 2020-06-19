import sys, json, torch
from data_util import TrainLoader, TestLoader
from util import LossClock, create_logger, max_margin_loss, mrr, ndcg, mapp, p
from model import HARM_Model


# Hyper-parameters
total_epochs = 20
lr = 2.0
k = 20  # 取top-k个
n = 9  # 测试集中查询个数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_file = "dump/vocab.json"
glove_emb_file = "dump/glove.emb"
model_dump_file = "dump/model.pth"
results_file = "output/results.json"

train_file = "data/event_train.csv"
dev_file = "data/event_dev.csv"
test_file = "test_event/event_rank_"

mode = sys.argv[1]
if mode != "train" and mode != "test":
    raise ValueError

# Running
with open(vocab_file, 'r') as f:
    vocab = json.load(f)
glove_emb = torch.load(glove_emb_file)
model = HARM_Model(len(vocab), glove_emb).to(device)

if mode == "train":
    best_mrr = 0.
    logger = create_logger("log/", "train.log")

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    clock = LossClock(["loss"], interval=20)

    ds_train = TrainLoader(train_file, vocab, device)
    ds_dev = TestLoader(dev_file, vocab, device)

    for epoch in range(total_epochs):
        # train
        logger.info("=" * 30 + f"Train epoch {epoch}" + "=" * 30)
        for query, docs in ds_train():
            r = model(query, docs)
            margin_loss = max_margin_loss(r[:1].expand(r[1:].size(0)), r[1:])
            # update
            optimizer.zero_grad()
            margin_loss.backward()
            optimizer.step()
            clock.update({"loss": margin_loss.item()})

        # evaluate
        logger.info("=" * 30 + f"Evaluate epoch {epoch}" + "=" * 30)
        rs, ls = [], []
        with torch.no_grad():
            for query, docs, label, _, _ in ds_dev():  # 用验证集进行评估
                r = model(query, docs)
                rs.append(r)
                ls.append(label)

        mrr_score = mapp(ls, rs)  # 评价指标
        if mrr_score > best_mrr:
            logger.info(f"Saving... ({mrr_score} > {best_mrr})")
            torch.save(model.state_dict(), model_dump_file)  # 保存模型
            best_mrr = mrr_score
        else:
            logger.info(f"Skip. ({mrr_score} < {best_mrr})")


map_list, p_list, ndcg_list = [], [], []
avg_map, avg_p, avg_ndcg = 0, 0, 0
if mode == "test":
    for i in range(1, n):  # n个查询
        path = test_file + str(i) + '.csv'
        ds_test = TestLoader(path, vocab, device)
        model.load_state_dict(torch.load(model_dump_file))

        ds_test = TestLoader(path, vocab, device)

        results, rs, ls = [], [], []
        with torch.no_grad():
            for query, docs, label, qid, doc_ids in ds_test():
                r = model(query, docs)
                rs.append(r)
                ls.append(label)

        p_score = p(rs, ls, k)  # 评价指标
        map_score = mapp(rs, ls)
        ndcg_score = ndcg(rs, ls, k)
        print('query' + str(i) + ':')
        print('precision:', p_score)
        p_list.append(p_score)

        print('map:', map_score)
        map_list.append(map_score)

        print('ndcg:', ndcg_score)
        ndcg_list.append(ndcg_score)

    print('avg_map:')
    print(sum(map_list) / n)
    print('avg_p:')
    print(sum(p_list) / n)
    print('avg_ndcg:')
    print(sum(ndcg_list) / n)