### README

In the code, you will find the following section in execute.py:

```python
#"Cornell", "Pubmed", "Cora", "Disease", "Citeseer", "Telecom", "Airport", "Actor" 
#5  2  1  6  3  7  8  4
for step, (data1, data2, data3, data4, data5, data6) in enumerate(
    zip(loader1, loader6, loader3, loader7, loader8, loader4)
):
```

For different downstream datasets, you need to adjust the six upstream datasets inside `enumerate`.

* If the downstream dataset is **Pubmed** or **Cornell**, no adjustment is required.
* If the downstream dataset is **Cora** or **Disease**, modify the order to:

```python
5 2 3 7 8 4
```



To run this code, run:
```
python execute.py
```