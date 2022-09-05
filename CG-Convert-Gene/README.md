# Convert-Gene

## Env Preparing
```
git clone https://github.com/IlikeBB/Convert-Gene.git
```
```
python setup.py
```

## Data Preparing
> * input data must csv type.
```
LOC, Gene
1231, C
4513, G
21432, G
.
.
.
```

## Start Convert
> * python convert.py --input *.csv --output {file_name}

```
python convert.py --input demo/feature_list.csv --output test --nullchange False
```

##  Results
<img src='https://github.com/IlikeBB/Convert-Gene/blob/main/demo/result2.png'></p>

