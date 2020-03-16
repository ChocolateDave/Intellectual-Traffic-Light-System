# Intellectual-Traffic-Light-System
## Introduction
This is a NUIETP sponsored project by Administration of Education, PRC. The main purpose of this project is to construct a robust and intelligent traffic light control system, with state-of-art machine learning techniques.

All rights of this project are reserved by college of Transportation Engineering at Tongji University and college of Software Engineering at Tongji University. Special thanks to German Aerospace Centre (DLR) for their generous support on our team.

>
>### College of Transportation Engineering at Tongji University
>### College of Software Engineering at Tongji University
>### German Aerospace Centre
>
>Website: www.tongji.edu.cn
>
>Team: 
>
>| Title               | Name | Homepage                                 |
>| ------------------- | ---- | ---------------------------------------- |
>| Professor | Xuesong Wang  | [http://www.tjsafety.cn/MembersInformation.aspx?YNID=487&YNID2=334&ID=495](http://www.tjsafety.cn/MembersInformation.aspx?YNID=487&YNID2=334&ID=495) |
>| Bachelor              | Juanwu Lu  | [https://github.com/ChocolateDave](https://github.com/ChocolateDave) |
>| Bachelor              | Lei Chen 
>| Bachelor              | Xiaonan Shi
## File Structure
```
.
├── sumo      // Traffic simulation environment based on Sumo (DLR)
│   ├── ITLS.add.xml
│   ├── ITLS.net.xml
│   ├── ITLS.rou.xml
│   ├── ITLS.sumo.cfg
│   ├── sumo.log.xml
│   └── view.settings.xml
├── agent.py        // Reinforcement learning agent
├── config.py       // Configurations
├── core.py          // Main function for training and evaluation 
├── env.py           // Reinforcement learning environment
└── memory.py   //  Reinforcement learning memory buffer
```

## Project configuration
### 0. Requirements
### 1. Training
### 2. Benchmark

## References
[[1] Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015). https://doi.org/10.1038/nature14236](https://www.nature.com/articles/nature14236)  
[[2] Wei, Hua & Zheng, Guanjie & Yao, Huaxiu & Li, Zhenhui. (2018). IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control. 2496-2505. 10.1145/3219819.3220096. ](https://www.researchgate.net/publication/326504263_IntelliLight_A_Reinforcement_Learning_Approach_for_Intelligent_Traffic_Light_Control)  
[[3] Xuesong, W., Linjia, H., Meixin, Z., Chai, C. Calibrating Car-Following Models on Surface Roads Using Shanghai Naturalistic Driving Data. In: 98th Annual Meeting of the Transportation Research Board. Washington DC, United States.](https://trid.trb.org/view/1573124)  
[[4] X. Liang, X. Du, G. Wang and Z. Han, "A Deep Reinforcement Learning Network for Traffic Light Cycle Control," in IEEE Transactions on Vehicular Technology, vol. 68, no. 2, pp. 1243-1253, Feb. 2019.](https://ieeexplore.ieee.org/document/8600382)