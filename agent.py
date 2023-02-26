import torch
import torch.nn as nn
from torchviz import make_dot


class network(nn.Module):
    def __init__(self,input_dim,linear_size1,hidden_size,linear_size2,output_dim,activation=None) -> None:
        super(network,self).__init__()

        self.ls1=linear_size1
        self.ls2=linear_size2
        self.hs=hidden_size
        self.acs=output_dim


        self.dense1=nn.Linear(input_dim,linear_size1)
        self.dense2=nn.Linear(6*hidden_size,output_dim)


        self.memory_1=nn.LSTM(linear_size1,hidden_size,batch_first=True)
        self.memory_2=nn.LSTM(linear_size1,hidden_size,batch_first=True)
        self.memory_3=nn.LSTM(linear_size1,hidden_size,batch_first=True)
        self.memory_4=nn.LSTM(linear_size1,hidden_size,batch_first=True)
        self.memory_5=nn.LSTM(linear_size1,hidden_size,batch_first=True)
        
        self.com1=nn.LSTM(linear_size1,hidden_size,batch_first=True)
        self.com2=nn.LSTM(linear_size1,hidden_size,batch_first=True)
        self.com3=nn.LSTM(linear_size1,hidden_size,batch_first=True)
        self.com4=nn.LSTM(linear_size1,hidden_size,batch_first=True)
        self.com5=nn.LSTM(linear_size1,hidden_size,batch_first=True)

    def forward(self,input,h_0,c_0): # input is shape (bs,num,obs_size), h_0,c_0=(num,1,bs,hidden_size)

        # LAYER 1
        ob1=self.dense1(input[:,0,:]).view(-1,1,self.ls1) #shape = (bs,1,linear_size1)
        ob2=self.dense1(input[:,1,:]).view(-1,1,self.ls1)
        ob3=self.dense1(input[:,2,:]).view(-1,1,self.ls1)
        ob4=self.dense1(input[:,3,:]).view(-1,1,self.ls1)
        ob5=self.dense1(input[:,4,:]).view(-1,1,self.ls1)

        #LAYER 2
        #SELF MEMORY
        mem1,(h1_n,c1_n)=self.memory_1(ob1,(h_0[0],c_0[0])) #shape is mem=(bs,num,hidden_size) h_n,c_n=(1,bs,hidden_size)
        mem2,(h2_n,c2_n)=self.memory_2(ob2,(h_0[1],c_0[1]))
        mem3,(h3_n,c3_n)=self.memory_3(ob3,(h_0[2],c_0[2]))
        mem4,(h4_n,c4_n)=self.memory_4(ob4,(h_0[3],c_0[3]))
        mem5,(h5_n,c5_n)=self.memory_5(ob5,(h_0[4],c_0[4]))

        ob1=ob1.view(-1,self.ls1)
        ob2=ob2.view(-1,self.ls1)
        ob3=ob3.view(-1,self.ls1)
        ob4=ob4.view(-1,self.ls1)
        ob5=ob5.view(-1,self.ls1)

        h_n=torch.stack([h1_n,h2_n,h3_n,h4_n,h5_n])
        c_n=torch.stack([c1_n,c2_n,c3_n,c4_n,c5_n]) #shape =(num,1,bs,hidden_size)

        #COMMUNICATION LAYER
        com1_input=torch.stack([ob2,ob3,ob4,ob5,ob1],dim=1) #shape = (bs,num,linear_1)
        com2_input=torch.stack([ob1,ob3,ob4,ob5,ob2],dim=1)
        com3_input=torch.stack([ob1,ob2,ob4,ob5,ob3],dim=1)
        com4_input=torch.stack([ob1,ob2,ob3,ob5,ob4],dim=1)
        com5_input=torch.stack([ob1,ob2,ob3,ob4,ob5],dim=1)


        com1=self.com1(com1_input)[0] #shape = (bs,num,hidden_size)
        com2=self.com2(com2_input)[0]
        com3=self.com3(com3_input)[0]
        com4=self.com4(com4_input)[0]
        com5=self.com5(com5_input)[0]

        #LAYER 3
        out1_input=torch.cat([com1[:,4],com2[:,0],com3[:,0],com4[:,0],com5[:,0],mem1[:,-1]],dim=1) # shape is (bs, (num_agent+1)*hidden_size )
        out2_input=torch.cat([com1[:,0],com2[:,4],com3[:,1],com4[:,1],com5[:,1],mem2[:,-1]],dim=1)
        out3_input=torch.cat([com1[:,1],com2[:,1],com3[:,4],com4[:,2],com5[:,2],mem3[:,-1]],dim=1)
        out4_input=torch.cat([com1[:,2],com2[:,2],com3[:,2],com4[:,4],com5[:,3],mem4[:,-1]],dim=1)
        out5_input=torch.cat([com1[:,3],com2[:,3],com3[:,3],com4[:,3],com5[:,4],mem5[:,-1]],dim=1)

        out1=self.dense2(out1_input) #shape = (bs,action_size)
        out2=self.dense2(out2_input)
        out3=self.dense2(out3_input)
        out4=self.dense2(out4_input)
        out5=self.dense2(out5_input)

        return out1,out2,out3,out4,out5,h_n,c_n


class FCMNet(nn.Module):
    def __init__(self,obs_size,ls1,hs,ls2,action_size) -> None:
        super(FCMNet,self).__init__()

        self.obs=obs_size
        self.ls1=ls1
        self.hs=hs
        self.ls2=ls2
        self.acs=action_size

        self.actor=network(obs_size,ls1,hs,ls2,action_size)
        self.critic=network(obs_size,ls1,hs,ls2,1)
