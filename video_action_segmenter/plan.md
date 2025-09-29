 ## Core Idea
 - 输入：完整的长视频 or 实时相机视频流
 - 处理：
    - Motion Tokenizer -> latent matrix -> energy
 - 输出：
    - 描绘一条能量曲线
    - 根据能量曲线进行动作分割（确认动作发生和结束的时间，输出为时间戳，并分割后的视频片段）

### Detailed Plan
