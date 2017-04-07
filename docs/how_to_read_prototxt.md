## solver
solver parameterについて
net : 学習用のprotoファイル名 テストネットと一緒でも良い
net_interbal: ２つのテストphase間の繰り返し回数
base_lr: ベース学習レート
momentum: モーメント値
weight_decay: 重み付けの遅延
clip_gradioents:0.1 ??????????
lr_policy:学習レートの遅延ポリシー step
gamma : 学習レートのガンマ
stepsize 学習レートのポリシーのステップのステップサイズ
display 情報を表示する間隔（繰り返しの回数）、０なら表示しない
max_iter 繰り返しの最大数（上限）
snapshot スナップショットを取る間隔 [default = 0]（多分０だと取らない？）
snapshot_prefix  スナップショットファイルの置き場所（ファイル名prefix、文字列）
solver_mode solverがGPUかCPUか選択 [default = GPU] （このパラメタの値は文字列ではなく、enum型{CPU=0, GPU=1}


## layer
name  この層の名前（文字列）
type 層のタイプ（下記のenumのうち１つ）
top 上側のblobの名前（文字列） 基本的にtop = name にする
bottom 下側のblobの名前（文字列） 受け取るデータ
hdf5_data_param HDF5（階層的データをまとめたファイル形式）のパラメタ。ソースとバッチサイズを指定
include この層がネットにいつ（どの状態stateで）取り込まれるかの指定
convolution_param ConvolutionParameter（下記）の値

enum LayerType {
  NONE = 0;
  ABSVAL = 35;
  ACCURACY = 1;
  ARGMAX = 30;
  BNLL = 2;
  CONCAT = 3;
  CONTRASTIVE_LOSS = 37;
  CONVOLUTION = 4;
  DATA = 5;
  DROPOUT = 6;
  DUMMY_DATA = 32;
  EUCLIDEAN_LOSS = 7;
  ELTWISE = 25;
  FLATTEN = 8;
  HDF5_DATA = 9;
  HDF5_OUTPUT = 10;
  HINGE_LOSS = 28;
  IM2COL = 11;
  IMAGE_DATA = 12;
  INFOGAIN_LOSS = 13;
  INNER_PRODUCT = 14;
  LRN = 15;
  MEMORY_DATA = 29;
  MULTINOMIAL_LOGISTIC_LOSS = 16;
  MVN = 34;
  POOLING = 17;
  POWER = 26;
  RELU = 18;
  SIGMOID = 19;
  SIGMOID_CROSS_ENTROPY_LOSS = 27;
  SILENCE = 36;
  SOFTMAX = 20;
  SOFTMAX_LOSS = 21;
  SPLIT = 22;
  SLICE = 33;
  TANH = 23;
  WINDOW_DATA = 24;
  THRESHOLD = 31;
}

optional uint32 num_output = 1; // The number of outputs for the layer
 optional bool bias_term = 2 [default = true]; // whether to have bias terms
 // Pad, kernel size, and stride are all given as a single value for equal
 // dimensions in height and width or as Y, X pairs.
 optional uint32 pad = 3 [default = 0]; // The padding size (equal in Y, X)
 optional uint32 pad_h = 9 [default = 0]; // The padding height
 optional uint32 pad_w = 10 [default = 0]; // The padding width
 optional uint32 kernel_size = 4; // The kernel size (square)
 optional uint32 kernel_h = 11; // The kernel height
 optional uint32 kernel_w = 12; // The kernel width
 optional uint32 group = 5 [default = 1]; // The group size for group conv
 optional uint32 stride = 6 [default = 1]; // The stride (equal in Y, X)
 optional uint32 stride_h = 13; // The stride height
 optional uint32 stride_w = 14; // The stride width
 optional FillerParameter weight_filler = 7; // The filler for the weight
 optional FillerParameter bias_filler = 8; // The filler for the bias
 enum Engine {
   DEFAULT = 0;
   CAFFE = 1;
   CUDNN = 2;
 }
 optional Engine engine = 15 [default = DEFAULT];
