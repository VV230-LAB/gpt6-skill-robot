# GPT6 Skill Robot

这个项目在 RoboDojo 仿真中研究一个问题：能否把重复的几何计算和运动推进交给程序，减少 GPT 介入，同时保持操作效果？

这里提供**可在 CPU 上独立运行的相机几何工具**，以及原生 GPT 控制、上下文交接、有界执行和局部学生策略的实验摘要。整套仿真、GPT 连接和训练环境需要原研究工作区。

## 快速开始

在仓库根目录运行。验证使用 Python 3.11。静态检查与测试均在本机 CPU 上运行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

python -B -m tools.initialize \
  --config tools/configs/robodojo-x5-nominal.json --config-only

python -B -m unittest discover -s tools/tests -v
```

第一条工具命令应返回 `CONFIG_READY`，表示名义配置完整。测试使用合成相机观测，覆盖投影、三角测量、当前请求绑定和缺项处理。实际对接 RoboDojo 时，在已确认的观测边界传入最新公开请求、图像与实测末端执行器（EEF）状态。完整命令和 Python 接口见[几何工具手册](tools/README.md)。

## 设计

| 层 | 职责 |
| --- | --- |
| 持续 GPT 会话 | 从当前图像选择对象、部位、目标和观察节点，判断接触、持物与释放效果 |
| 宿主与上下文 | 跟踪当前观测、已确认动作和剩余额度，必要时用 C2 把当前状态、近期证据和文件版本交给新线程 |
| 几何工具 | 按当前请求绑定相机模型和实测 EEF，提供投影、射线、三角测量和显式平面求交 |
| EPC 程序化执行 | 在一次授权内逐步推进目标，复用原控制器，并在观察节点交回实际执行量、误差和图像 |

原生 R 是持续会话中由 GPT 逐段给出 EEF 动作的对照。EPC 由程序展开已授权目标。C2 负责上下文接续。三者的职责与边界见[方法与接口](docs/methods.md)。

## 实验结果一览

| 实验 | 观察结果 |
| --- | --- |
| 堆碗，EPR1 | 同一固定配置下原生成功 **5/5**，原图均支持套叠、释放与撤离，成本和时间联合达标 **4/5**。平均 **422 万 reported tokens／723 秒**，中层承担 **2210/2347** 个确认步（ACK） |
| 堆积木，EPC-BLK1 | 同期一对均成功。EPC 组 reported tokens 少 **65.34%**，完整墙钟少 **7.90%** |
| 上下文融合，MPE1 | 两次堆碗完成，四次真实换线程均接续动作。平均 **501 万／1102 秒**，高于历史 EPR1 均值 |
| 跨任务，CORE2 与 CLS1 | CORE2 完成一次普通档堆积木。有界运动也在交换积木和分类中执行，两项均未取得完整成功的成本对照 |
| 局部学生，Correction BC GRU | 停止头补训使两组在 FULL 测试的正确停止从 **0/4 到 2/4**。B 组 SHORT 从 **4/4 降到 2/4**，整体未通过不退化门 |

这些是固定配置下的观测。EPR1 使用历史 Fast 档，普通档 STD2 是另一次成功堆碗参照。Reported tokens 含缓存输入，实际费用未知。原生任务成功与严格归位分别记录。逐条数字、参照和失败范围见[实验结果](docs/results.md)与[运行摘要 CSV](docs/runs.csv)。

## 目录

- [`tools/`](tools/README.md) 包含只读 CLI、Python 接口、配置和 CPU 测试。
- [`docs/methods.md`](docs/methods.md) 说明当前状态、工具复用、有界执行和停止判定。
- [`docs/results.md`](docs/results.md) 说明历史实验、口径、限制和原始研究索引。

几何工具基于 [RoboDojo](https://github.com/robodojo-benchmark/RoboDojo) 与 [GPT-as-Policy](https://github.com/anonymous-report-421/GPT-as-Policy) 的公开接口与名义相机关系，版本见[来源说明](docs/methods.md#来源与配置)。本仓库的发布许可待项目所有者确定。
