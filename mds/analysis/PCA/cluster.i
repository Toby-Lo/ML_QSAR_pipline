parm complex.parm7
trajin Name_combine.nc
readdata projection.txt name pca_data

# K-means clustering on the PCA data
cluster data pca_data kmeans clusters 2 \
        out kmeans.clusters \
        summary kmeans.summary \
        rms :1-194@CA mass
run

#python3 PCA.py projection.txt PCA pca.png
