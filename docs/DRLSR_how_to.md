Agisoftを積んだCMUくんでの計算方法メモ

基本的には以下のディレクトリにファイルがある。
```
F:\caffe-windows\examples\DRLSR
```

### データセットの生成方法
```
data_aug.m
generate_test.m
generate_train.m
```

### caffe関連ファイル
- Matlab用ファイル
DRLSR_clip_mat_inception_v2.prototxt
- ネットワークセットアップ用ファイル
DRLSR_clip_net_inception_v2.prototxt
- 環境セットアップ用ファイル
DRLSR_clip_solver_inception_v2.prototxt

### ネットワーク学習の実行方法
```
F:\caffe-windows>caffe train -solver examples\DRLSR\DRLSR_clip_solver_final_v2.prototxt
```

### スナップショットから続ける方法
```
F:\caffe-windows>caffe train -solver examples\DRLSR\DRLSR_clip_solver_final_v2.prototxt -snapshot examples\DRLSR\DRLSR_v2_tuning_iter_198.solverstate
```

### Matlab形式のモデル保存方法
```
saveFiltersDRLSRinception.m
```

### テスト実行方法
```
F:\caffe-windows\examples\DRLSR\Testing_stage
demo_DRLSR.m
```
