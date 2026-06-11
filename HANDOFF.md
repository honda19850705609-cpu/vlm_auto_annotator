# VLM Auto-Annotator —— 对话交接文档(HANDOFF)

> 给下一个对话:这是 VisDrone "VLM 自动标注 → 检测器蒸馏" 项目的完整状态。
> 读完这份就能无缝接上。日期 ~2026-06-11。

---

## 0. 一句话项目

用 **Qwen2.5-VL 零样本自动标注** VisDrone(航拍密集小目标),量化质量、想办法逼近人工标注,
并验证"VLM 伪标注能训出反超 VLM 的检测器"。**核心矛盾:小目标(尤其 <8px)是 VLM 的硬墙。**

## 1. 环境 / 机器 / 仓库(关键!)

- **代码仓库**:Mac 桌面 `~/Desktop/GitHub/vlm_auto_annotator`(在这改 + push)。
  GitHub: `https://github.com/honda19850705609-cpu/vlm_auto_annotator`(分支 main)。
  **用户偏好:文件只建在 ~/Desktop/GitHub,不要建在 Google Drive 副本里。**
- **计算在 5090**(AutoDL,`connect.westc.seetacloud.com`)。**我(助手)无法 SSH,只能给命令让用户跑、贴回结果。**
  - 5090 上仓库:`/root/autodl-tmp/Day11/vlm_auto_annotator`(最新 clone);老的在 Day10/Day9。
  - **GFW 坑**:`git pull`/`raw.githubusercontent` 常卡。解法:`source /etc/network_turbo`(AutoDL 学术加速,每个新终端都要 source);或用 jsDelivr `https://cdn.jsdelivr.net/gh/honda19850705609-cpu/vlm_auto_annotator@main/<file>`。
  - **长任务**:`PYTHONUNBUFFERED=1 nohup python -u ... &`(默认 nohup 缓冲,日志看着像卡住但其实在跑)。输出 JSON 只在跑完才写出。
  - **脚本要在仓库目录里跑**(`python xx.py` 找当前目录);数据用绝对路径。常用:`REPO=$(dirname $(find /root/autodl-tmp/Day11 -name badcase.py 2>/dev/null|head -1)); cd "$REPO"`。

## 2. 数据路径(5090)

- val 图(109 张):`/root/autodl-tmp/visdrone_val_gt109/`
- val 真值 COCO:`/root/autodl-tmp/Day9/vlm_auto_annotator/visdrone_val_gt.json`(109 图 / 8076 框,10 类细分;badcase 会归一到 3 核心类 pedestrian/vehicle/bicycle)
- 模型权重:`/root/autodl-tmp/qwen2.5-vl-7b/`(主力)、`qwen2.5-vl-3b/`
- 蒸馏数据:`/root/autodl-tmp/Day10/distill/`(train300/ 300张训练图、train300_gt.json 真值、pseudo*.json、ds_*/ YOLO集、runs/ 训练结果、det_*_coco.json、bc_*/)
- 当前 SAM 实验:`/root/autodl-tmp/Day11/anno/`(smoke5/ 5张最难的图、gt5.json、各 *_coco.json、bc_*/)
- `yolov8s.pt`:在 5090 某处(`find /root -name yolov8s.pt`);训练时 `model=` 用绝对路径,别让它去 GitHub 下(会卡)。

**5 张最难的 smoke 图**:`0000129_02411_d_0000138 / 0000244_02000_d_0000005 / 0000291_00001_d_0000868 / 0000295_01800_d_0000030 / 0000295_02000_d_0000031`(`.jpg`)。gt5 共 1103 核心框。

## 3. 评估口径(badcase.py)

GT 锚定、纯 CPU、按 **file_name 对齐**、自由词标签归一到 3 核心类、每图每类贪心 IoU≥0.5。
输出总体/分类别 P/R/F1 + **按目标尺寸分档召回**(<8/8-16/16-32/32-96/≥96,按 sqrt(area))。小目标召回是命门。

---

## 4. 已完成的结果(全部已 push)

### 4a. VLM 标注质量(全量 109 图)
| 方法 | P | R | F1 |
|---|---|---|---|
| 整图 VLM | 0.538 | 0.045 | 0.082 |
| 多尺度切图 v3 (640:1+512:2) | 0.544 | 0.243 | 0.336 |
| v4(修复空图 prompt) | 0.439 | **0.420** | 0.430 |
| **v4 + 去幻觉(最好的 VLM 标注)** | **0.509** | **0.420** | **0.460** |

v4+dehall 分尺寸召回:<8 **0.075**, 8-16 0.187, 16-32 0.404, 32-96 0.692, ≥96 0.859。

**关键修复**:空图 bug——旧 prompt 在低分辨率图(如 540×960)上返回 `[]`,导致 23/109 图全空(占 11.5% GT)。改成"召回优先"prompt 后召回 0.243→0.420。**精度靠结构性后处理(白名单 normalize_label + dehallucinate.py 去等距网格幻觉)守,不靠 prompt 死压。**

### 4b. 蒸馏(300 train 图伪标注 → YOLOv8s,eval on val 109)
| | VLM老师 | D0 伪标注 | D1 SAHI自训练 | 人工天花板 |
|---|---|---|---|---|
| F1 | 0.460 | **0.505** | **0.535** | 0.673 |
| P | 0.509 | 0.764 | 0.608 | 0.715 |
| R | 0.420 | 0.377 | 0.477 | 0.636 |
| 速度 | ~100s/图 | ~2ms/图 | ~2ms/图 | ~2ms/图 |

- **D0 反超 VLM 老师**(F1 0.505>0.460,精度 0.764 翻倍,快~1万倍),且"去噪"(训练标签 P=0.50 → 检测器 P=0.76)。
- **D1**:SAHI 切图自标补全标签(标签召回 0.41→0.53,小目标翻倍)→ 重训,F1 0.505→0.535,**恢复人工天花板 79.5%**,全程零人工。
- D1 分尺寸召回:<8 0.103, 8-16 0.258, 16-32 0.473, 32-96 0.740, ≥96 0.805。
- 第 2 轮自训练(D2)**饱和**:300 图上检测器回收自己的错误(确认偏置),标签 F1 没再涨。**结论:再提升要么加数据,要么换范式。**

### 4c. Day10 副结论
- **VLM 自报置信度无校准**,无法当筛选刀(最佳 F1 在 thr~0.6 = 几乎不滤)。
- 去幻觉:全量 v3 上 P 0.544→0.579(去 215 FP / 只丢 2 TP)。

---

## 5. 当前进行中:换范式"解耦定位+分类"(在 5 张 smoke 图上验证)

**动机**:v4/v5/SR 全卡在 <8px(切图/超分都撬不动)。诊断:逼 VLM 干它最弱的"在降采样图里定位极小目标"。**新范式:类无关定位器(SAM)出框 → VLM 只做分类(它的强项)。**

### 5 张 smoke 图上的完整消融(gt5=1103 框)
| 方法 | P | R | F1 | <8 | 8-16 |
|---|---|---|---|---|---|
| v4 基线 | 0.399 | 0.296 | **0.340** | 0.039 | 0.137 |
| v5(384:3细尺度+2048token) | 0.244 | 0.391 | 0.300 | 0.054 | 0.187 |
| v5+去幻觉 | 0.306 | 0.386 | 0.342 | 0.054 | 0.180 |
| SR超分+切图 | 0.257 | 0.295 | 0.275 | 0.031 | 0.101 |
| **SAM 类无关定位(只看召回)** | — | **0.598** | — | **0.186** | **0.428** |
| SAM+CLIP 分类 | 0.094 | 0.519 | 0.159 | 0.116 | 0.337 |
| SAM+CLIP 调阈值最佳 | 0.250 | 0.372 | 0.299 | — | — |
| SAM+Qwen montage(grid5) | 0.065 | 0.209 | 0.099 | 0.039 | 0.142 |

**关键发现:**
1. **SR 反而更差**——超分让每瓦片可见小目标暴增 → token 截断把"看见的"又丢了(每块~50个截断)。<8px 是 VLM 物理边界(降采样+token顶)。
2. **SAM 撞穿了定位墙**:类无关定位召回 0.598,<8px **0.186(VLM 的 3.4×)**!证明"墙是 VLM 定位能力的墙,不是数据的墙"。但 SAM 高召回低精度(6496 候选,大量背景)。
3. **分类器成了新瓶颈**:CLIP 太弱(分不清小航拍 crop 的目标/背景,精度 0.09,调阈值最高 F1 0.299)。Qwen montage(一次 25 个)**对齐崩溃**——它数不全25个标签,后面补 none → 召回暴跌 0.519→0.209。

### ⏳ 下一步(用户即将跑,尚无结果)
**`classify_proposals_vlm.py --grid 3`**(一次只 9 个 crop,Qwen 能数全排准)——验证 montage 失败是"太挤"还是"Qwen 真分不动":
```bash
W=/root/autodl-tmp/Day11/anno
python classify_proposals_vlm.py --prop $W/sam_prop.json --images $W/smoke5 \
  --model /root/autodl-tmp/qwen2.5-vl-7b --out $W/sam_qwen3_coco.json \
  --grid 3 --cell 128 --context 2.5 --max-new-tokens 256
python badcase.py --gt $W/gt5.json --vlm $W/sam_qwen3_coco.json --out $W/bc_samqwen3
grep -E "总体|<8|8-16|16-32|32-96" $W/bc_samqwen3/report.md
```
- 召回回到 ~0.5 且精度上来 → montage 太挤,范式活了,继续优化 + 推全量。
- 召回还低 → Qwen 也分不动极小 crop → 分类是新墙,记录边界。
- (SAM 候选 `sam_prop.json` 已生成在 Day11/anno;SAM 用 vit_b,`sam_propose.py --tile 640 --points-per-side 48`。CLIP 分类用 `classify_proposals.py`(open_clip)。)

---

## 6. 仓库脚本清单(都在 GitHub)

| 脚本 | 作用 |
|---|---|
| `minimal_vlm` / `structured_vlm` | VLM 推理基础 / 批量结构化标注(JSON 框) |
| `tiled_vlm` | 多尺度切图标注(核心);`--scales 640:1.0,512:2.0` `--upscale` `--max-box-frac`;含召回优先 VISDRONE_PROMPT + 白名单 |
| `to_coco` | annotations.json → COCO;`--box-scale`(SR 坐标缩回) |
| `badcase` | GT 锚定评估(P/R/F1 + 分尺寸召回 + badcase排序);`--pred` 三方+rescue率 |
| `dehallucinate` | 去等距网格幻觉(提精度) |
| `analyze_confidence` | 置信度阈值扫描 → PR曲线/最佳F1 |
| `diag_tile` | 空图诊断(隔离 prompt vs 切图) |
| `coco_to_yolo` / `yolo_to_coco` | 蒸馏:COCO↔YOLO;`yolo_to_coco --tile` 支持 SAHI 推理(整图+切块融合) |
| `sahi_relabel` | 检测器 SAHI 切图自标 + 融合 VLM(自训练补全标签) |
| `make_tile_dataset` | 瓦片原生训练集(超纲·检测器侧,留档) |
| `sr_preprocess` | 超分预处理(Real-ESRGAN/lanczos)——**已验证无效** |
| `sam_propose` | SAM everything 类无关候选框 |
| `agnostic_recall` | 类无关定位召回(只看定位不看类) |
| `classify_proposals` | CLIP 给 SAM 候选分类(弱) |
| `classify_proposals_vlm` | Qwen-VL montage 给 SAM 候选分类(进行中,调 grid) |
| `devlog/day9-13` `distill_runbook` `results/` | 全部日志 + 结果归档 |

---

## 7. 概念结论(论文/答辩用,用户很看重这块)

1. **VLM vs 检测器标注的本质**:检测器更准更快,但**前提是已有人工标注**(否则训不出来,鸡生蛋死循环)。VLM 唯一不可替代价值 = **冷启动**:零人工标注、开放词表,从无到有造标签。"用训好的 DINO 标注"是伪命题——它偷用了训练时的人工标注。
2. **为什么 VLM 能而检测器不能(虽然都是模型)**:差别不在模型,在**训练数据的成本经济学**。VLM 的标注成本被"预付"在几十亿张免费网络图文上(通用、社区出资、一次性),你白嫖;检测器的标注成本每个任务都要你现付昂贵专家框。**VLM=继承的通用知识(广而浅)→ 零成本冷启动;检测器=购买的专精知识(窄而深)→ 强但要先付钱。**
3. **VLM 的关键 = 标注能力的"边际成本"≈0**(零样本/免标注/免训练),代价是**单次推理慢 + 质量广而浅**。所以小目标(VisDrone 这种细分分布,VLM 预训练里罕见)天然弱 = <8px 墙的根源。
4. **完整流水线**:VLM(标注成本≈0,中等质量)当冷启动标注器 → 检测器(高质量高速度,但要标签)当最终产品;检测器消化 VLM 标注后**反超** VLM。整个项目就是优化这条"大模型→检测器"的标注线。
5. **建议论文 intro 论证主线**(用户想要):① VLM=标注器/检测器=产品的分工 → ② 通用知识 vs 专精知识、预付 vs 现付成本 → ③ 标注成本 vs 标注质量的权衡。三段连起来 = "为什么非 VLM 不可"的完整立论。

## 8. 开放的下一步(按价值)
- **(进行中)** SAM+Qwen grid=3:救不救得活解耦范式的分类瓶颈。
- 若解耦成:推全量 109,得到新范式的 headline。
- 若解耦不成:坐实"定位可被 SAM 破,但极小目标分类是新墙",收口为诚实边界结论。
- 备选(标注侧扩展,检测器固定):伪标注 300→1000 张,看"零人工@量 vs 人工@质"。
- 用户明确要的待办:把第 7 节三段框架写成**论文 intro 论证草稿**。

## 9. 用户风格 / 注意
- 中文交流;喜欢一步步跑实验、贴结果、要诚实判读(别浮夸,负结果也如实说)。
- 在意"是不是跑偏"——区分**标注任务(方向二,他的核心)** vs **检测器优化(方向一,超纲)**。改 YOLO 模型(P2头/瓦片训练)被他判为超纲。
- 每轮重要成果都要 **commit + push 到 GitHub**;数据文件被 .gitignore 挡(别提交大 json)。
- 给命令时:绝对路径、先 cd 到仓库、`source /etc/network_turbo`、长任务用 nohup -u。
