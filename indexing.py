#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


a=np.array([1,2,3,4,5],ndmin=2)


# In[3]:


a.ndim


# In[4]:


a.dtype


# In[5]:


b=a.astype(str)


# In[6]:


b.dtype


# In[7]:


b.nbytes


# In[8]:


a.shape


# In[9]:


d=np.array([[1,2,3],[4,5,6]])


# In[10]:


d.shape


# In[11]:


x=np.array([1,2,3,4,5,6,7,8,9,10,11,12])


# In[12]:


b=x.reshape(2,2,3)


# In[13]:


b


# In[14]:


e=np.zeros((2,2),dtype=int)


# In[15]:


e


# f=np.zeros((9,9),dtype=int)

# In[16]:


f=np.zeros((9,9),dtype=int)


# In[17]:


f


# In[178]:


f[0:8,2]


# In[19]:


f[1::2, ::2] = 1


# In[20]:


f


# In[21]:


f[::2, 1::2] = 1


# In[22]:


f


# In[23]:


g=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])


# In[24]:


g


# In[25]:


g[2,1]


# In[26]:


g[2,1:5]


# In[27]:


g[1:4,2:5]


# In[28]:


g[::3,0:6]


# In[29]:


g[::3,2]


# In[30]:


g[::2,::2]


# In[31]:


g[::2,:1]


# In[32]:


h=np.zeros((9,9),dtype=int)


# In[33]:


h[1::2, ::2] = 1


# In[34]:


h


# In[35]:


h[::2, 1::2] = 1


# In[36]:


h


# In[37]:


g[1::2,:1]


# In[38]:


i=np.zeros((9,9),dtype=int)


# In[39]:


i[0:10,::2]=1


# In[40]:


i


# In[41]:


j=np.zeros((9,9),dtype=int)


# In[42]:


j[1::2,::2]=1


# In[43]:


j


# In[44]:


j[::2,1::2]=1


# In[45]:


j


# In[46]:


x=np.full(4,6)


# In[47]:


x


# In[48]:


y=np.full((2,2),3)


# In[49]:


y


# In[50]:


o=np.full([5,5],1,dtype=float)


# In[51]:


o


# In[52]:


o[1:4,1:4]=0


# In[53]:


o


# In[54]:


o[2,2:3]=9


# In[55]:


o


# In[56]:


u=np.linspace(0,100,5)


# In[57]:


u


# In[58]:


v=np.arange(1,20)


# In[59]:


v


# In[60]:


w=np.arange(1,20,3)


# In[61]:


w


# In[62]:


l=np.arange(1,13)


# In[63]:


l


# In[64]:


l1=l.reshape(2,2,3)


# In[65]:


l1


# In[66]:


l.size


# In[67]:


l=np.random.random(5)


# In[68]:


l


# In[69]:


y=np.random.randint(4,9,2)


# In[70]:


y


# In[71]:


q=np.random.randint(4,9,(2,2))


# In[72]:


q


# In[73]:


np.append(q,7)


# In[74]:


x=np.array([1,2,3])


# In[75]:


z=np.array([4,5,6])


# In[76]:


np.append(x,z)


# In[77]:


np.insert(x,1,2)


# In[78]:


r=np.array([1,2,3,4,5])


# In[79]:


r


# In[80]:


for i in range (0,5):
    print(r[i])


# In[81]:


r1=np.array([[1,2,3],[3,4,5]])


# In[82]:


r1.ndim


# In[83]:


for i in range (0,2):
    for j in range(0,3):
        print(r1[i][j])


# In[84]:


r2=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])


# In[ ]:





# In[85]:


for i in range (0,2):
    for j in range(0,2):
        for k in range(0,3):
            print(r2[i][j][k])


# In[86]:


u=np.array([[1,2,3],[4,5,6],[7,8,9]])


# In[87]:


y=[100,78,56]


# In[88]:


s=np.insert(x,1,z,axiz=0)


# In[89]:


ab=np.array([1,6,3,2])


# In[90]:


abc=np.copy(ab)


# In[91]:


abc


# In[ ]:





# In[92]:


np.sort(ab)


# In[93]:


np.sort(r2,axis=0)


# In[94]:


np.sort(r2,axis=1)


# In[95]:


a=np.array([4,5,6,7])


# In[96]:


a*2


# In[97]:


a-2


# In[98]:


a+2


# In[99]:


a/2


# In[100]:


b=np.array([2,6,4,3])


# In[101]:


a+b


# In[102]:


a-b


# In[103]:


a*b


# In[104]:


a/b


# In[105]:


a


# In[106]:


b


# In[107]:


np.subtract(4,2)


# In[108]:


np.add(a,b)


# In[109]:


np.subtract(a,b)


# In[110]:


np.multiply(a,b)


# In[111]:


np.sqrt(a)


# In[112]:


np.sin(0)


# In[113]:


np.cos(0)


# In[114]:


np.log(10)


# In[115]:


np.log10(2)


# In[116]:


np.exp(5)


# In[117]:


ar=np.array([[1,2,3,4,5],[6,7,8,9,10]])


# In[118]:


arr=np.array([[1,2,3,4,5],[6,7,8,9,10]])


# In[119]:


ar+arr


# In[120]:


ar-arr


# In[121]:


ar*arr


# In[122]:


ar/arr


# In[123]:


ar%arr


# In[124]:


np.concatenate((ar,arr))


# In[125]:


np.vstack([ar,arr])


# In[126]:


np.concatenate((ar,arr),axis=1)


# In[127]:


np.hstack([ar,arr])


# In[128]:


np.ravel(arr)


# In[129]:


arr.reshape(-1)


# In[130]:


a=np.ones((2,3),dtype=int)


# In[131]:


b=np.full((3,2),5)


# In[132]:


a


# In[133]:


b


# In[134]:


c=np.matmul(a,b)


# In[135]:


c


# In[136]:


#satistics


# In[137]:


abc=np.array([[1,2,3],[2,5,7],[3,6,9]])


# In[138]:


abc


# In[139]:


np.max(abc)


# In[140]:


np.max(abc,axis=1)


# In[141]:


np.max(abc,axis=0)


# In[142]:


abc.min()


# In[143]:


np.min(abc,axis=0)


# In[144]:


np.min(abc,axis=1)


# In[145]:


arr


# In[146]:


np.sum([arr,ar])


# In[147]:


np.sum([arr,ar],axis=0)


# In[148]:


np.sum([arr,ar],axis=1)


# In[149]:


np.mean(arr)


# In[150]:


np.median(arr)


# In[151]:


np.std(arr)


# In[152]:


np.var(arr)#variance


# In[153]:


r1=np.array([[1,2,3],[3,4,5]])


# In[154]:


bool_arr = np.ones((3,3), dtype=bool)


# In[155]:


bool_arr


# In[156]:


w=np.linspace(10,100,25,dtype=int)


# In[157]:


w


# In[158]:


w[w%2==1]=-1


# In[159]:


w


# In[160]:


a1=np.array([1,2,3,4,5])


# In[161]:


a2=np.array([5,6,7,8,9,10])


# In[162]:


a2[[2,4,5]]


# In[163]:


import os


# In[164]:


data=np.genfromtxt('juyptor123.txt',delimiter=',')


# In[165]:


a=np.array([[1,2,3],[4,5,6],[7,8,9]])


# In[166]:


np.split(a,3,0)


# In[167]:


np.split(a,3,1)


# In[168]:


np.split(a,[2,3],0)


# In[169]:


a=np.array([1,2,3])


# In[170]:


b=np.array([2,6,4])


# In[171]:


np.equal(a,b)


# In[173]:


a2


# In[174]:


g=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25],[26,27,28,29,30]])


# In[175]:


g


# In[176]:


g[0,3:5]


# In[177]:


import pandas as pd


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




