# data handling
import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn

# get NDCG
from torch.autograd import Variable
import math
import heapq

# train model
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from time import time

from tqdm import tqdm

'''
可使用gpu训练
'''
if torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cpu')
'''
第一阶段导入数据，
和数据处理，将数据处理成tensor张量

'''


class GDataset(object):

    def __init__(self, user_path, group_path, num_negatives):
        '''
        Constructor
        '''
        self.num_negatives = num_negatives
        # user data
        self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + "Train.txt")
        self.user_testRatings = self.load_rating_file_as_list(user_path + "Test.txt")
        self.user_testNegatives = self.load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_items = self.user_trainMatrix.shape
        # group data
        self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + "Train.txt")
        self.group_testRatings = self.load_rating_file_as_list(group_path + "Test.txt")
        self.group_testNegatives = self.load_negative_file(group_path + "Negative.txt")


    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1:]:
                    if x =='\n':
                        continue
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    def get_user_dataloader(self, batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_group_dataloader(self, batch_size):
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        print("train data", train_data)
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return group_train_loader




# 建立Agree主体模型

class GRbOL(nn.Module):
    def __init__(self, num_users, num_items, num_groups, embedding_dim, group_member_dict, drop_ratio):
        super(GRbOL, self).__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim).to(device)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.MoSANattention_WCmembers = SubUserInteractionAttention_Layer_C(embedding_dim, drop_ratio)
        self.MoSANattention_WUmembers = SubUserInteractionAttention_Layer_U(embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, group_inputs, user_inputs, item_inputs):
        # train group
        if (group_inputs is not None) and (user_inputs is None):
            out = self.grp_forward(group_inputs, item_inputs)
        # train user
        else:
            out = self.usr_forward(user_inputs, item_inputs)
        return out

    # group forward
    def grp_forward(self, group_inputs, item_inputs):
   
        group_embeds = self.groupembeds(torch.LongTensor(group_inputs))
        item_embeds_full = self.itemembeds(torch.LongTensor(item_inputs))
        group_ContextUser_userLatent = torch.zeros(32)
        #此处获得一个群组中，影响力最大的用户的下标索引。
       
        R_groups_items_list =[]
        
        for i, j in zip(group_inputs, item_inputs):
            #print("group_ inputs ", group_inputs)
            #print("i.item()", i.item())
            group = i.item()
            if group not in self.group_member_dict.keys():
                members = self.group_member_dict[150]
            else: 
                members = self.group_member_dict[i.item()]
            ## initialize members_embeds from cpu to gpu
            members_embeds = self.userembeds(torch.LongTensor(members))
            items_numb = []
            size = len(members)
            
            for _ in members:
                items_numb.append(j)
                
                
         
            ###  initialize item_embeds from cpu to gpu
            item_embeds = self.itemembeds(torch.LongTensor(items_numb))
            
            
            
            group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)
            at_wt = self.attention(group_item_embeds)
            """
               at_wt : 代表群组不同用户的影响权重。获得群组意见领袖下的群组意见得分
            """
            pred, idx = at_wt[0].max(0)
            leader_user_index = idx
            group_ContextUser_userLatent += self.getGilm(leader_user_index, members_embeds,
                                                         size)  # group_ContextUser_userLatent tensor 32*1
            Recommend_group_toItem = torch.matmul(group_ContextUser_userLatent.t(), item_embeds[0])
            R_groups_items_list.append([abs(Recommend_group_toItem)])
            R_groups_items_list_tensor = torch.Tensor(R_groups_items_list)
            
            
            #AGREE模型用户权重，乘以用户向量....修改后的模型无需使用
            #g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
            ###  initialize group_embeds_pure from cpu to gpu
            #group_embeds_pure = self.groupembeds(torch.LongTensor([i]))
            #g_embeds = g_embeds_with_attention + group_embeds_pure
            #group_embeds = torch.cat((group_embeds, g_embeds))

        element_embeds = torch.mul(group_embeds, item_embeds_full)  # Element-wise product
        new_embeds = torch.cat((element_embeds, group_embeds, item_embeds_full), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))+R_groups_items_list_tensor
        #print('format a = {}'.format(a))#group score y  tensor([[0.4938],[0.3169],[0.3567],...,[0.5862]], device='cuda:0', grad_fn=<SigmoidBackward>)
        #print('format r = {}'.format(R_groups_items_list_tensor))
        #print('format y = {}'.format(y))
        return y

    

 # user forward
    def usr_forward(self, user_inputs, item_inputs):
        #print(" user _inputs cuda or not ", user_inputs)# not
        #print(" item _inputs cuda or not ", item_inputs)# 
        ###  from cpu to gpu
        user_embeds = self.userembeds(user_inputs)
        item_embeds = self.itemembeds(item_inputs)
        element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product
        new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)
        y = torch.sigmoid(self.predictlayer(new_embeds))
        return y

    # get g(i l m)
    def getGilm(self, context_user_index, members_embeds, size):
        '''

        :param context_user_embed: ci
        :param latent_user_embed:  um
        :return: g(i m)
        '''
        context_user_embed = members_embeds[context_user_index]  # get context user's embed  32*1
        context_user_embed = context_user_embed.to(device)
        latent_user_embeds = members_embeds[torch.arange(members_embeds.size(0)) != context_user_index]
        latent_user_embeds = latent_user_embeds.to(device)
        at_wt_MoSAN_context = self.MoSANattention_WCmembers(context_user_embed)
        at_wt_MoSAN_latent_list = self.MoSANattention_WUmembers(latent_user_embeds)  # 使用同一个网络， 节省资源
        latentUserEmbeds_with_attation_list = [i + at_wt_MoSAN_context for i in at_wt_MoSAN_latent_list]

        g_ilm = latent_user_embeds[0] * 0
        temp = latent_user_embeds
        for i in range(size - 1):
            temp[i] = latentUserEmbeds_with_attation_list[0][0][i] * latent_user_embeds[i]
            g_ilm = g_ilm + temp[i]
            g_ilm = g_ilm
        g_ilm = (g_ilm + context_user_embed)/2
        return g_ilm
    
    
    
class UserEmbeddingLayer(nn.Module):
    '''
    torch.nn.Embedding
    一个简单的查找表（lookup table），存储固定字典和大小的词嵌入。
    此模块通常用于存储单词嵌入并使用索引检索它们(类似数组)。模块的输入是一个索引列表，输出是相应的词嵌入。
    :param
        num_embeddings - 词嵌入字典大小，即一个字典里要有多少个词。
        embedding_dim - 每个词嵌入向量的大小。
    :return  最终返回一个一个用户所代表的词嵌入向量

    '''

    def __init__(self, num_users, embedding_dim):
        # 字典大小为user的人数，每一个词嵌入向量的大小为预先设好的embedding_dim
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_inputs):
        user_embeds = self.userEmbedding(user_inputs)
        ## from cpu to gpu
        
        return user_embeds


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


class GroupEmbeddingLayer(nn.Module):
    def __init__(self, number_group, embedding_dim):
        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim)

    def forward(self, num_group):
        group_embeds = self.groupEmbedding(num_group)
        return group_embeds


class AttentionLayer(nn.Module):
    '''
    A sequential container. Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    :param
    :return:
    '''

    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            # dropout 防止过拟合
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = torch.softmax(out.view(1, -1), dim=1)
        return weight
    
    
    
class SubUserInteractionAttention_Layer_C(nn.Module):
    '''
    based on paper Interact and Decide: Medley of Sub-Attention Networks for
    Effective Group Recommendation

    aimed to train a attention layer for each group,
    problems:  need to create much layer, and need to store the config lists include all layers



    :param
    :return:
    '''

    def __init__(self, embedding_dim, drop_ratio=0):
        super(SubUserInteractionAttention_Layer_C, self).__init__()
        # Linear 32*16
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            # dropout 防止过拟合
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        out = out
        # torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)，同时将out的维度变为一行n列的向量
        weight = torch.softmax(out.view(1, -1), dim=1)
        return weight
    
class SubUserInteractionAttention_Layer_U(nn.Module):
    '''
    based on paper Interact and Decide: Medley of Sub-Attention Networks for
    Effective Group Recommendation

    aimed to train a attention layer for each group,
    problems:  need to create much layer, and need to store the config lists include all layers



    :param
    :return:
    '''

    def __init__(self, embedding_dim, drop_ratio=0):
        super(SubUserInteractionAttention_Layer_U, self).__init__()
        # Linear 32*16
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            # dropout 防止过拟合
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        # torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)，同时将out的维度变为一行n列的向量
        weight = torch.softmax(out.view(1, -1), dim=1)
        return weight

class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out



# 进行测试

class Helper(object):
    """

    """

    def __init__(self):
        self.timber = True

    def gen_group_member_dict(self, path):
        group_member_dict = {}
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                array = line.split(' ')
                group = int(array[0])
                group_member_dict[group] = []
                for m in array[1].split(','):
                    group_member_dict[group].append(int(m))
                line = f.readline().strip()
        return group_member_dict

    def evaluate_model(self, model, testRatings, testNegatives, K, type_m):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        HiT_Ratio, NDCG = [], []

        for idx in range(len(testRatings)):
            (hit_ratio, ndcg) = self.eval_one_rating(model, testRatings, testNegatives, K, type_m, idx)
            HiT_Ratio.append(hit_ratio)
            NDCG.append(ndcg)
        return (HiT_Ratio, NDCG)

    def eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx):
        rating = testRatings[idx]
        items = testNegatives[idx]
        user = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), user)

        users_var = torch.from_numpy(users)
        ##   from cpu to gpu
        users_var = users_var.long()
        ## from cpu to gpu
        items_var = torch.LongTensor(items)
        if type_m == 'group':
            predictions = model(users_var, None, items_var)
        elif type_m == 'user':
            predictions = model(None, users_var, items_var)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.data.numpy()[i]
            #print('map_score item is ',item," item score is ",map_item_score[item] )                           
        items.pop()
                                  

        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i + 2)
        return 0


# train the model
def training(model, train_loader, epoch_id, config, type_m):
    # user trainning
    learning_rates = config.lr
    # learning rate decay
    lr = learning_rates[0]
    if epoch_id >= 15 and epoch_id < 25:
        lr = learning_rates[1]
    elif epoch_id >=20:
        lr = learning_rates[2]
    # lr decay
    if epoch_id % 5 == 0:
        lr /= 2

    # optimizer
    optimizer = optim.RMSprop(model.parameters(), lr)

    losses = []
    print('%s train_loader length: %d' % (type_m, len(train_loader)))
    for batch_id, (u, pi_ni) in tqdm(enumerate(train_loader)):
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0]
        neg_item_input = pi_ni[:, 1]
        # Forward
        if type_m == 'user':
            pos_prediction = model(None, user_input, pos_item_input)
            neg_prediction = model(None, user_input, neg_item_input)
        elif type_m == 'group':
            pos_prediction = model(user_input, None, pos_item_input)
            neg_prediction = model(user_input, None, neg_item_input)
        # Zero_grad
        model.zero_grad()
        # Loss
        loss = torch.mean((pos_prediction - neg_prediction -1) **2)
        # record loss history
        losses.append(loss)  
        # Backward
        loss.backward()
        optimizer.step()

    print('Iteration %d, loss is [%.4f ]' % (epoch_id, torch.mean(torch.stack(losses))))


def evaluation(model, helper, testRatings, testNegatives, K, type_m):
    model.eval()
    (hits, ndcgs) = helper.evaluate_model(model, testRatings, testNegatives, K, type_m)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    return hr, ndcg


class Config(object):
    def __init__(self):
        self.path = '../input/mafengmo/SoAGREE_MaFengwo/'
        self.user_dataset = self.path + 'userRating'
        self.group_dataset = self.path + 'groupRating'
        self.user_in_group_path = "../input/mafengmo/SoAGREE_MaFengwo/groupMember.txt"
        self.embedding_size = 32
        self.epoch = 30#30
        self.num_negatives = 32
        self.batch_size = 32   # 256
        self.lr = [0.000005, 0.000001, 0.0000005]
        self.drop_ratio = 0.2
        self.topK = 5
        
if __name__ == '__main__':
    # initial parameter class
    config = Config()

    # initial helper
    helper = Helper()

    # get the dict of users in group
    g_m_d = helper.gen_group_member_dict(config.user_in_group_path)

    # initial dataSet class
    dataset = GDataset(config.user_dataset, config.group_dataset, config.num_negatives)

    # get group number
    num_group = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items

    # build AGREE model
    grbol = GRbOL(num_users, num_items, num_group, config.embedding_size, g_m_d, config.drop_ratio)

    # config information
    print("AGREE at embedding size %d, run Iteration:%d, NDCG and HR at %d" %(config.embedding_size, config.epoch, config.topK))
    # train the model
    for epoch in range(config.epoch):
        grbol.train()
        # 开始训练时间
        t1 = time()
        #training(grbol, dataset.get_user_dataloader(config.batch_size), epoch, config, 'user')

        training(grbol, dataset.get_group_dataloader(config.batch_size), epoch, config, 'group')
        print("user and group training time is: [%.1f s]" % (time()-t1))
        # evaluation
        t2 = time()
        #u_hr, u_ndcg = evaluation(agree, helper, dataset.user_testRatings, dataset.user_testNegatives, config.topK, 'user')
        #print('User Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]' % (
            #epoch, time() - t1, u_hr, u_ndcg, time() - t2))

        hr, ndcg = evaluation(grbol, helper, dataset.group_testRatings, dataset.group_testNegatives, config.topK, 'group')
        print(
            'Group Iteration %d [%.1f s]: HR = %.4f, '
            'NDCG = %.4f, [%.1f s]' % (epoch, time() - t1, hr, ndcg, time() - t2))


    print("Done!")
