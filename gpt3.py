import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import numpy as np
import time
from tqdm import tqdm 
with open('C:/Users/17710/Desktop/data/train.txt',encoding='utf-8')as f:
	lines=f.readlines()
	f.close()
trains_data=[]
temp_data=''
for line in lines:
	if line !='\n':
		line=line.strip()
		temp_data+=(line+'\t')
	else:
		trains_data.append(temp_data)
		temp_data=''
print(trains_data[0:3])
train_data=[]
for data in trains_data:
	if len(data)<300:
		train_data.append(data)
train_data=train_data[::2]
word_count={}
print(trains_data[0:2])
for data in train_data:
	data=data.strip().replace('\t','')
	for word in data:
		word_count.setdefault(word,0)
		word_count[word]+=1
word2id={"<pad>":0,"<unk>":1,"<sep>":2}
temp={word:i+len(word2id)for i,word in enumerate(word_count.keys())}
word2id.update(temp)

print(word2id['你'])
id2word=list(word2id.keys())
device=torch.device('cuda:0'if torch.cuda.is_available()else 'cpu')
vocab_size=len(word2id)
max_pos=300
d_model=768
d_ff=2048
d_k=d_v=64
n_layers=6
n_heads=8
CLIP=1
def make_data(datas):
	train_datas=[]
	for data in datas:
		data=data.strip()
		train=[i if i!='\t'else "<sep>" for i in data]+['<sep>']
		train_datas.append(train)
	return train_datas

class Data(Dataset):
	def __init__(self,datas):
		super().__init__()
		self.datas=datas 
	def __getitem__(self,item):
		data=self.datas[item]
		inputs=data[:-1]
		targets=data[1:]
		inputs_len=len(inputs)
		targets_len=len(targets)
		return{"inputs":inputs,"inputs_len":inputs_len,
		"targets":targets,"targets_len":targets_len}

	def __len__(self):
		return len(self.datas)

	def padding_batch(self,batch):
		input_len=[d["inputs_len"]for d in batch]
		target_len=[d["targets_len"]for d in batch]
		input_max=max(input_len)
		target_max=max(target_len)
		for d in batch:
			d["inputs"].extend([word2id["<pad>"]]*(input_max-d['inputs_len']))
			d["targets"].extend([word2id["<pad>"]]*(target_max-d['targets_len']))
		decoder_inputs=torch.tensor([d["inputs"]for d in batch],dtype=torch.long)
		decoder_targets=torch.tensor([d["targets"]for d in batch],dtype=torch.long)
		return decoder_inputs,decoder_targets 
def get_attn_pad_mask(seq_q,seq_k):
	batch_size,len_q=seq_q.size()
	batch_size,len_k=seq_k.size()
	pad_mask=seq_k.data.eq(0).unsqueeze(1)
	return pad_mask.expand(batch_size,len_q,len_k)

def get_attn_subsequence_mask(seq):
	shape=[seq.size(0),seq.size(1),seq.size(1)]
	mask=np.triu(np.ones(shape),k=1)
	mask=torch.from_numpy(mask).byte()
	mask=mask.to(device)
	return mask

class ScaledDotProductAttention(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self,Q,K,V,attn_mask):
		scores=torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
		scores.masked_fill_(attn_mask,-1e9)
		attn=nn.Softmax(dim=-1)(scores)
		context=torch.matmul(attn,V)
		return context,attn  

class MultiHeadAttention(nn.Module):
	def __init__(self):
		super().__init__()
		self.W_Q=nn.Linear(d_model,d_k*n_heads,bias=False)
		self.W_K=nn.Linear(d_model,d_k*n_heads,bias=False)
		self.W_V=nn.Linear(d_model,d_v*n_heads,bias=False)
		self.fc=nn.Linear(n_heads*d_v,d_model,bias=False)
		self.layernorm=nn.LayerNorm(d_model)

	def forward(self,input_Q,input_K,input_V,attn_mask):
		residual,batch_size=input_Q,input_Q.size(0)
		Q=self.W_Q(input_Q).view(batch_size,-1,n_heads,d_k).transpose(1,2)
		K=self.W_K(input_K).view(batch_size,-1,n_heads,d_k).transpose(1,2)
		V=self.W_V(input_V).view(batch_size,-1,n_heads,d_v).transpose(1,2)
		attn_mask=attn_mask.unsqueeze(1).repeat(1,n_heads,1,1)
		context,attn=ScaledDotProductAttention()(Q,K,V,attn_mask)
		context=context.transpose(1,2).reshape(batch_size,-1,n_heads*d_v)
		output=self.fc(context)
		return self.layernorm(output+residual),attn

class PoswiseFeedForwardNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc=nn.Sequential(
			nn.Linear(d_model,d_ff,bias=False),
			nn.ReLU(),
			nn.Linear(d_ff,d_model,bias=False))
		self.layernorm=nn.LayerNorm(d_model)
	def forward(self,inputs):
		residual=inputs
		output=self.fc(inputs)
		return self.layernorm(output+residual)
class DecoderLayer(nn.Module):
	def __init__(self):
		super().__init__()
		self.dec_self_attn=MultiHeadAttention()
		self.dec_enc_attn=MultiHeadAttention()
		self.pos_ffn=PoswiseFeedForwardNet()

	def forward(self,dec_inputs,dec_self_attn_mask):
		dec_outputs,dec_self_attn=self.dec_self_attn(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
		dec_outputs=self.pos_ffn(dec_outputs)
		return dec_outputs,dec_self_attn  

class Decoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.tgt_emb=nn.Embedding(vocab_size,d_model)
		self.pos_emb=nn.Embedding(max_pos,d_model)
		self.layers=nn.ModuleList([DecoderLayer()for _ in range(n_layers)])

	def forward(self,dec_inputs):
		seq_len=dec_inputs.size(1)
		pos = torch.arange(seq_len,dtype=torch.long,device=device)
		pos=pos.unsqueeze(0).expand_as(dec_inputs)
		dec_outputs=self.tgt_emb(dec_inputs)+self.pos_emb(pos)
		dec_self_attn_pad_mask=get_attn_pad_mask(dec_inputs,dec_inputs)
		dec_self_attn_subsequence_mask=get_attn_subsequence_mask(dec_inputs)
		dec_self_attn_mask=torch.gt((dec_self_attn_pad_mask+dec_self_attn_subsequence_mask),0)
		dec_self_attns=[]
		for layer in self.layers:
			dec_outputs,dec_self_attn=layer(dec_outputs,dec_self_attn_mask)
			dec_self_attns.append(dec_self_attn)

		return dec_outputs,dec_self_attns

class GPT(nn.Module):
	def __init__(self):
		super().__init__()
		self.decoder=Decoder()
		self.projection=nn.Linear(d_model,vocab_size)

	def forward(self,dec_inputs):
		dec_outputs,dec_self_attns=self.decoder(dec_inputs)
		dec_logits=self.projection(dec_outputs)
		return dec_logits.view(-1,dec_logits.size(-1)),dec_self_attns 

	def greedy_decoder(self,dec_input):
		terminal=False
		start_dec_len=len(dec_input[0])
		while not terminal:
			if len(dec_input[0])-start_dec_len>100:
				next_symbol=word2id['<sep>']
				dec_input=torch.cat(
					[dec_input.detach(),torch.tensor([[next_symbol]],dtype=dec_input.dtype,device=device)],-1)
				break
			dec_outputs,_=self.decoder(dec_input)
			projected=self.projection(dec_outputs)
			prob=projected.squeeze(0).max(dim=-1,keepdim=False)[1]
			next_word=prob.data[-1]
			next_symbol=next_word
			if next_symbol==word2id["<sep>"]:
				terminal=True
			dec_input=torch.cat(
				[dec_input.detach(),torch.tensor([[next_symbol]],dtype=dec_input.dtype,device=device)],-1)
		return dec_input 

	def answer(self,sentence):
		dec_input=[word2id.get(word,1)if word!='\t'else word2id['<sep>']for word in sentence]
		dec_input=torch.tensor(dec_input,dtype=torch.long,device=device).unsqueeze(0)
		output=self.greedy_decoder(dec_input).squeeze(0)
		out=[id2word[int(id)]for id in output]
		sep_indexes=[]
		for i in range(len(out)):
			if out[i]=="<sep>":
				sep_indexes.append(i)
		answer=out[sep_indexes[-1]+1:-1]
		answer="".join(answer)
		return answer

def epoch_time(start_time,end_time):
	elapsed_time=end_time-start_time
	elapsed_mins=int(elapsed_time/60)
	elapsed_secs=int(elapsed_time)
	return elapsed_mins,elapsed_secs

def train_step(model,data_loader,optimizer,criterion,clip=1,print_every=None):
	model.train()
	if print_every==0:
		print_every=1
	print_loss_total=0
	epoch_loss=0
	for i,(dec_inputs,dec_outputs)in enumerate(tqdm(data_loader)):
		optimizer.zero_grad()
		dec_inputs,dec_outputs=dec_inputs.to(device),dec_outputs.to(device)
		outputs,dec_self_attns=model(dec_inputs)
		loss=criterion(outputs,dec_outputs.view(-1))
		print_loss_total+=loss.item()
		epoch_loss+=loss.item()
		loss.backward()
		torch.nn.utils.clip_grad_norm(model.parameters(),clip)
		optimizer.step()
		if print_every and(i+1)%print_every==0:
			print_loss_avg=print_loss_total/print_every
			print_loss_total=0
			print('\tcurrent_loss:%.4f'%print_loss_avg)
	return epoch_loss/len(data_loader)


def train(model,data_loader):
	criterion=nn.CrossEntropyLoss(ignore_index=0).to(device)
	optimizer=optim.Adam(model.parameters(),lr=1e-4)
	for epoch in range(epochs):
		start_time=time.time()
		train_loss=train_step(model,data_loader,optimizer,criterion,CLIP,print_every=10)
		end_time=time.time()
		torch.save(model.state_dict(),'GPT3.pt')
		epoch_mins,epoch_secs=epoch_time(start_time,end_time)
		print(epoch,epoch_mins,epoch_secs)
		print(f'\tTrain Loss:{train_loss: .3f}')

def print_num_parameters(model):
	total_params=sum(p.numel()for p in model.parameters())
	print(total_params)
	total_trainable_params=sum(p.numel()for p in model.parameters()if p.requires_grad)
	print(f'{total_trainable_params:,}training parameters')

if __name__=='__main__':
	input_data=make_data(train_data)
	input_data_num=[[word2id[word]for word in line]for line in input_data]
	batch_size=2
	epochs=30
	dataset=Data(input_data_num)
	loader=DataLoader(dataset,batch_size=batch_size,collate_fn=dataset.padding_batch)
	model=GPT().to(device)
	train(model,loader)


	





