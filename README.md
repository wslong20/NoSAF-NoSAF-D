## NoSAF-NoSAF-D

> Motivation: Different nodes may have different contributions to the deep layer.
>
> ![image](https://github.com/wslong20/NoSAF-NoSAF-D/assets/103408498/c75ffc59-c907-40e7-b1df-634857928b78)
> 
> ![image](https://github.com/wslong20/NoSAF-NoSAF-D/assets/103408498/24fce978-9756-444d-bd7b-8d6a200516b1)



### Experimental Results

![image](https://github.com/wslong20/NoSAF-NoSAF-D/assets/103408498/7f1edd3b-da52-4fc6-becb-b6fad643f217)



### Usage
#### Recommended Requirements

* torch=1.13.1+cu117

* torch-geometric=2.3.1

* ogb=1.3.6

* tqdm=4.65.0

* h5py=3.9.0

To train NoSAF on all small datasets, run:
```
chmod u+x NoSAF.sh
```
```
./NoSAF.sh
```

To train NoSAF-D on all small datasets, run:
```
chmod u+x DeepNoSAF.sh
```
```
./DeepNoSAF.sh
```

#### To train NoSAF or NoSAF-D on a single data set, you can find the corresponding command in the corresponding .sh file

