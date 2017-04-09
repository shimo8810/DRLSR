# DRLSR(Deep Residual Learning Super-resolution)
超解像深層学習研究
Hayato SHIMODAIRA, Muhammad HARIS, Hajime NOBUHARA.

## ToDo
- [x] 論文を読む
- [ ] 現環境のwin+caffe+matlabをlinux+tensorflow(or chainer)+python(and c++)に書き換える。無理なら頑張るしか無いよ
    - [x] データセットの準備(HDF5?)
        - [x] data argumentation
        - [ ] HDF5化
    - [x] gradient clipping実装
    - [x] ネットワークの書き換え(Chainer)
    - [ ] ネットワークの書き換え(TensorFlow)
    - [ ] 一般画像に対応するデモの作成
    - [ ] ハリスさんの再現実験
- [ ] 気になるところの改善 bagfix

## ディレクトリ構成
- docs
ドキュメントディレクトリ
- legacy
ハリス様の遺した遺産
- legacy/src
ハリス様の遺した遺産(コードの一部)

## データセット
現状既存のデータ・セットをnpy形式に保存し逐次呼び出している。
