# multiagent
3A2の講義「マルチエージェントシステム」の課題置き場

# やること
Sphere関数、Rosenbrock関数、Rastrign関数、Griewank関数、Alpine関数、2^n minima関数を粒子群最適化、および人工蜂コロニーによる最適化を行う

# 環境

Python 3, numpy, matplotlibが必要 

# 利用方法

コマンドラインで以下を実行

```
$ git clone git@github.com:ymaquarium/multiagent.git
$ cd multiagent/main/
$ python3 pso.py N S T dir_name
$ python3 abc.py N S T dir_name
```

Nは次元数、Sはサンプル数、Tは更新回数、dir_nameはresults/にて保存される場所の名前(新規作成もしくは上書き)


