# Agent 启动提示词

在 Cursor **Agent** 模式聊天中粘贴：

```text
/cuda-accelerate

任务：优化 agent_workdir 中的 NAFNet 端到端模型。skill中的目标是对某个算子或者融合的算子进行优化，你现在的任务是替换model.py中的某些节点并定制实现，具体的算子实现参考skill规定。这里的节点可以是融合的操作算子，你也需要分析如何融合最优，以及考虑到数据排布等，达到综合端到端模型最优。现在的目标是较torch.compile快10%，而不是5%。verification/profiling 中 baseline 与 torch.compile（model.py）均使用 fp16 精度评测。

约定：
- 工作目录：agent_workdir/
- baseline 已在 model.py（勿改）
- model_new.py 你在这里实现，定制实现通过torch_extension替换
- kernels/ 已清空，请从零实现 CUDA kernel + *_binding.cpp
- 不要改 utils/、binding.cpp、binding_registry.h、model.py
- 每次结果正确且有优化时，git 推送一次修改，并注明主要优化点

流程（严格按 skill）：
1. 在 kernels/ 实现涉及到的定制算子
2. 写 model_new.py：骨架与model.py一致，但定制算子使用extension替换
3. 编译：bash utils/compile.sh（按本机 GPU 设 TORCH_CUDA_ARCH_LIST）
4. 跑 python -m utils.verification，直到正确性通过
5. 跑 python -m utils.profiling，直到至少比 torch.compile 快 10%，并尽量继续压榨性能
6. 完成后清理 kernels/ 中间文件，只留最终实现

说明：verification/profiling 在 Windows 上一般不需要 sudo；若脚本写了 sudo 可去掉。
```
