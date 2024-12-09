# as3
use kvcache vs no 不量化 fp16 fp8对比
https://colab.research.google.com/drive/1kwtgn3Obsgf9KoQBy8Bh0WV3SIurwb7Y?usp=sharing
use kv cache后速度变快了，显存占用提高
关于量化结果很奇怪，每次跑结果都不一样，但是fp8显存占用应该最小，用的是colab t4
kv cache 实现结果
Time taken for golden greedy decoding without KV cache:  53.08241891860962
Time taken for customized greedy decoding:  8.432814359664917
在colab t4上结果
kv cache 实现结果
Time taken for golden greedy decoding without KV cache:  53.08241891860962
Time taken for customized greedy decoding:  8.432814359664917
在colab t4上结果

https://colab.research.google.com/drive/1uRtcSRFAeUQK9E7uYrfyPzwq29Q5vaLU#scrollTo=iFJ0RK1j0BHw&line=1&uniqifier=1 
reflection实现，在mbpp上达到0.6809338521400778准确率。
0.680933852140077
![image](https://github.com/user-attachments/assets/f8c4f107-1cef-42a1-9db5-e13b7166c05b)
reflection在gsm8k上准确率，
![image](https://github.com/user-attachments/assets/ba035f9a-f9bd-4c7e-a9dc-6bb37f0a68dd)
reflection在gsm8k表现不如cot

reflection gsm8k最后准确率为80左右低于cot的90左右
![Uploading image.png…]()
cot在mbpp上准确率0.72 高于reflection
![Uploading image.png…]()
icl最终实现
https://colab.research.google.com/drive/1OjBn0vYtSL3pAIxZXXKPs2C3F6sfu0oU?usp=sharing
reflection 
https://colab.research.google.com/drive/1uRtcSRFAeUQK9E7uYrfyPzwq29Q5vaLU?usp=sharing
