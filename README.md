# DRLSR(Deep Residual Learning Super-resolution)
超解像深層学習研究
Hayato SHIMODAIRA, Muhammad HARIS, Hajime NOBUHARA.

## ToDo
- [ ] 論文を読む
    - [x] ハリスさんの論文
    - [ ] Image Super-Resolution Using Deep COnvolutional Networks.(SRCNN)
    - [ ] Very Deep Convolutional Networks.(VDSR)
    - [ ] inception concept , GoogleNet
    - [ ] residual learning,
- [x] 現環境のwin+caffe+matlabをlinux+tensorflow(or chainer)+python(or c++)に書き換え
    - [x] データセットの準備(HDF5?)
        - [x] data argumentation
        - [ ] HDF5化
    - [x] gradient clipping実装
    - [x] ネットワークの書き換え(Chainer)
    - [ ] ネットワークの書き換え(TensorFlow)
    - [x] 一般画像に対応するデモの作成
    - [x] ハリスさんの再現実験
- [ ] 気になるところの改善 bagfix
- [ ] 出力結果ファイル群の調整
- [ ] 結果の改善（実験再現レベルまで）

## ディレクトリ構成
- docs
ドキュメントディレクトリ
- legacy
ハリス様の遺した遺産
- legacy/src
ハリス様の遺した遺産(コードの一部)

## データセット
現状既存のデータ・セットをnpy形式に保存し逐次呼び出している。
