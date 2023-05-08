## Computing the feature correspondences and the view order beforehand

0. Install [COLMAP](https://github.com/colmap/colmap) to ensure that you can call the ``colmap`` in your terminal.
1. Prepare the data in the following structure at ``data/{dataset}/{scene}``
```dotnetcli
data/ETH3D/meadow
├── images
│   ├── DSC_6535.jpg
│   ├── DSC_6536.jpg
│   ├── DSC_6537.jpg
│   ├── DSC_6538.jpg
│   ├── DSC_6539.jpg
│   ├── DSC_6540.jpg
│   ├── DSC_6541.jpg
│   ├── DSC_6547.jpg
│   ├── DSC_6548.jpg
│   ├── DSC_6553.jpg
│   ├── DSC_6556.jpg
│   ├── DSC_6557.jpg
│   ├── DSC_6558.jpg
│   ├── DSC_6559.jpg
│   └── DSC_6560.jpg
└── intrinsics.txt

```
2. Run the following command
```
python preparation/main.py --data ETH3D --scene meadow 
```

Then, the feature correspondences and the view order are saved as
```
data/ETH3D/meadow
├── database.db
├── images
│   ├── DSC_6535.jpg
│   ├── DSC_6536.jpg
│   ├── DSC_6537.jpg
│   ├── DSC_6538.jpg
│   ├── DSC_6539.jpg
│   ├── DSC_6540.jpg
│   ├── DSC_6541.jpg
│   ├── DSC_6547.jpg
│   ├── DSC_6548.jpg
│   ├── DSC_6553.jpg
│   ├── DSC_6556.jpg
│   ├── DSC_6557.jpg
│   ├── DSC_6558.jpg
│   ├── DSC_6559.jpg
│   └── DSC_6560.jpg
├── intrinsics.txt
├── log.txt
├── n_views.npy
├── pose
│   ├── DSC_6535.txt
│   ├── DSC_6536.txt
│   ├── DSC_6537.txt
│   ├── DSC_6538.txt
│   ├── DSC_6539.txt
│   ├── DSC_6540.txt
│   ├── DSC_6541.txt
│   ├── DSC_6547.txt
│   ├── DSC_6548.txt
│   ├── DSC_6553.txt
│   ├── DSC_6556.txt
│   ├── DSC_6557.txt
│   ├── DSC_6558.txt
│   ├── DSC_6559.txt
│   └── DSC_6560.txt
├── pose_graph.npy
├── sparse
│   └── 0
│       ├── cameras.bin
│       ├── images.bin
│       ├── points3D.bin
│       └── project.ini
└── two_view.npy
```
