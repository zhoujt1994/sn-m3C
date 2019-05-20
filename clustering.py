# embedding by mCG

rate_bin = np.load(outdir + 'matrix/cell_4238_rate.100kbin.mCG.npy')
read_bin = np.load(outdir + 'matrix/cell_4238_read.100kbin.mCG.npy')

binlist = []
data = []
for bat in range(3):
	for ind in indlist:
		cell = np.logical_and(batch==bat, indiv==ind)
		if np.sum(cell) > 0:
			rate_tmp = rate_bin[cell]
			read_tmp = read_bin[cell]
			s = np.sum(read_tmp>20, axis=0)
			binfilter = np.logical_and((s>=0.9*len(read_tmp)), bin_all[:,0]!='chrX')
			print(sum(cell), sum(binfilter))
			rateb = np.divide(rate_tmp[:, binfilter].T, meta[cell, 9].astype(float)).T
			readb = read_tmp[:, binfilter]
			for i in range(np.sum(binfilter)):
				rateb[readb[:,i]<20, i] = np.mean(rateb[readb[:,i]>=20, i])
			data.append(rateb)
			binlist.append(['-'.join(x.tolist()) for x in bin_all[binfilter]])

integrated, corrected, genes = scanorama.correct(data, binlist, return_dimred=True, hvg=None)
rateb_reduce = np.concatenate(integrated)
y = tsne.fit_transform(rateb_reduce[:, :ndim])

np.save(outdir + 'matrix/cell_4238_mCH_all_integrated_svd100.npy', rateb_reduce)
np.savetxt(outdir + 'matrix/cell_4238_mCH_all_integrated_svd50_p50_rs0.txt', y, delimiter='\t', fmt='%s')

# clustering by mCG

ndim = 50
#g1 = knn(np.load(outdir + 'matrix/cell_4238_mCG_all_integrated_svd100.npy')[:, :ndim], n_neighbors=20)
g = knn(np.load(outdir + 'matrix/cell_4238_mCH_all_integrated_svd100.npy')[:, :ndim], n_neighbors=20)
inter = g.dot(g.T)
diag = inter.diagonal()
jac = inter.astype(float) / (diag[None, :] + diag[:, None] - inter)
adj = nx.from_numpy_matrix(g.multiply(jac).toarray())
# adj = nx.from_numpy_matrix((g1+g2).toarray())
knnjaccluster = {}
for res in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0]:
	partition = community.best_partition(adj,resolution=res)
	label = np.array([k for k in partition.values()])
	knnjaccluster[res] = label
	nc = len(set(label))
	count = np.array([sum(label==i) for i in range(nc)])
	print(res, count)

np.save(outdir + 'matrix/cell_4238_mCH_all_integrated_svd50_knn20_louvain.npy', knnjaccluster)

# clustering neurons by mCH

neufilter = np.array([x in [0,1,3,4,6,8,12,13,14] for x in label])
rate_bin = np.load(outdir + 'matrix/cell_4238_rate.100kbin.mCH.npy')[neufilter]
read_bin = np.load(outdir + 'matrix/cell_4238_read.100kbin.mCH.npy')[neufilter]

binlist = []
data = []
neubatch = batch[neufilter]
neuindiv = indiv[neufilter]
neumeta = meta[neufilter]
for bat in range(3):
	for ind in indlist:
		cell = np.logical_and(neubatch==bat, neuindiv==ind)
		if np.sum(cell) > 0:
			rate_tmp = rate_bin[cell]
			read_tmp = read_bin[cell]
			s = np.sum(read_tmp>100, axis=0)
			binfilter = np.logical_and((s>=0.99*len(read_tmp)), bin_all[:,0]!='chrX')
			print(sum(cell), sum(binfilter))
			rateb = np.divide(rate_tmp[:, binfilter].T, neumeta[cell, 8].astype(float)).T
			readb = read_tmp[:, binfilter]
			for i in range(np.sum(binfilter)):
				rateb[readb[:,i]<100, i] = np.mean(rateb[readb[:,i]>=100, i])
			data.append(rateb)
			binlist.append(['-'.join(x.tolist()) for x in bin_all[binfilter]])

rate_tmp = np.load(outdir + '../PFC_pub/matrix/cell_2784_rate.100kbin.mCH.npy')
read_tmp = np.load(outdir + '../PFC_pub/matrix/cell_2784_read.100kbin.mCH.npy')
meta_tmp = np.load(outdir + '../PFC_pub/matrix/cell_2784_meta.npy')
s = np.sum(read_tmp>100, axis=0)
binfilter = np.logical_and((s>=0.99*len(read_tmp)), bin_all[:,0]!='chrX')
print(len(rate_tmp), sum(binfilter))
rateb = np.divide(rate_tmp[:, binfilter].T, meta_tmp[:, 8].astype(float)).T
readb = read_tmp[:, binfilter]
for i in range(np.sum(binfilter)):
	rateb[readb[:,i]<100, i] = np.mean(rateb[readb[:,i]>=100, i])

data.append(rateb)
binlist.append(['-'.join(x.tolist()) for x in bin_all[binfilter]])


integrated, corrected, genes = scanorama.correct(data, binlist, return_dimred=True, hvg=None)

rateb_reduce = np.concatenate(integrated)
y = tsne.fit_transform(rateb_reduce[:, :ndim])

np.save(outdir + '../PFC_pub/matrix/cell_4398_mCH_all_integrated_svd100.npy', rateb_reduce)
np.savetxt(outdir + '../PFC_pub/matrix/cell_4398_mCH_all_integrated_svd50_p50_rs0.txt', y, delimiter='\t', fmt='%s')

cluster = label.copy()
ndim = 50
g = knn(rateb_reduce[:, :ndim], n_neighbors=20)
inter = g.dot(g.T)
diag = inter.diagonal()
jac = inter.astype(float) / (diag[None, :] + diag[:, None] - inter)
adj = nx.from_numpy_matrix(g.multiply(jac).toarray())
# adj = nx.from_numpy_matrix((g1+g2).toarray())
knnjaccluster = {}
for res in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0]:
	partition = community.best_partition(adj,resolution=res)
	label = np.array([k for k in partition.values()])
	knnjaccluster[res] = label
	nc = len(set(label))
	count = np.array([sum(label==i) for i in range(nc)])
	print(res, count)

np.save(outdir + 'matrix/neu_1614_mCH_all_integrated_svd50_knn20_jac_louvain.npy', knnjaccluster)

neuy = y[neufilter]
for res in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0]:
	label = knnjaccluster[res]
	nc = len(set(label))
	count = np.array([sum(label==i) for i in range(nc)])
	tot = 0
	fig,ax = plt.subplots()
	ax.set_frame_on(False)
	ax.axis('off')
	for i in [2,5,7,9,10,11,15,16,17]:
		cell = (cluster==i)
		ax.scatter(y[cell,0],y[cell,1],s=8,c=color[i],alpha=0.8,edgecolors='none',label='cluster'+str(i+1))
		ax.text(np.median(y[cell,0]),np.median(y[cell,1]),str(i+1),fontsize=12,horizontalalignment='center',verticalalignment='center')
	for i in range(max(label)+1):
		if count[i]>5:
			cell = (label==i)
			ax.scatter(neuy[cell,0], neuy[cell,1], s=8, c=color[tot], alpha=0.8, edgecolors='none', label='cluster'+str(tot+1))
			ax.text(np.median(neuy[cell,0]),np.median(neuy[cell,1]),str(tot+1),fontsize=12,horizontalalignment='center',verticalalignment='center')
			tot+=1
	plt.legend(bbox_to_anchor=(-0.25,1), loc="upper left")
	plt.tight_layout()
	plt.savefig(outdir + 'plot/cell_4238_mCG_all_integrated_svd50_p50_rs0.mCG_all_integrated_svd50_knn20_jac_louvain_res1.6.neu_mCH_all_integrated_svd50_knn20_jac_louvain_res'+str(res)+'.pdf',bbox_inches="tight",transparent=True)
	plt.close()


pubbatch = np.array(neubatch.tolist() + [3 for i in range(len(meta_tmp))])
clustername = np.sort(list(set(meta_tmp[:,-1])))

label = np.load(outdir + 'matrix/neu_1614_mCH_all_integrated_svd50_knn20_jac_louvain.npy').item()[1.6]
nc = len(set(label))
selc = [[0,1,4,14],[10],[3,5,7],[6,16],[8,11,17],[12,18],[9,19],[2,13,15]]
neuleg = ['' for i in range(nc)]
leg = ['L2/3', 'L4', 'L5', 'L6', 'Ndnf', 'Vip', 'Pvalb', 'Sst']
for xx,yy in zip(selc,leg):
	for i,c in enumerate(xx):
		neuleg[c] = yy+'-'+str(i+1)

neulabel = np.array([neuleg[x] for x in label])

label = np.load(outdir + 'matrix/cell_4238_mCG_all_integrated_svd50_knn20_jac_louvain.npy').item()[1.6]
nc = len(set(label))
selc = [[0,1,3,4,6,8,12,13,14],[5],[2,9],[7],[11],[10],[15,16,17]]
allleg = ['' for i in range(nc)]
leg = ['Neuron', 'Astro', 'ODC', 'OPC', 'MG', 'MP', 'Endo']
for xx,yy in zip(selc,leg):
	for i,c in enumerate(xx):
		allleg[c] = yy+'-'+str(i+1)

alllabel = np.array([allleg[x] for x in label])

neumerge = np.array([x.split('-')[0] for x in neulabel])
allmerge = np.array([x.split('-')[0] for x in alllabel])

cluster = allmerge.copy()
cluster[allmerge=='Neuron'] = neumerge.copy()

subcluster = alllabel.copy()
subcluster[allmerge=='Neuron'] = neulabel.copy()


# embedding by mCH

rate_bin = np.load(outdir + 'matrix/cell_4238_rate.100kbin.mCH.npy')
read_bin = np.load(outdir + 'matrix/cell_4238_read.100kbin.mCH.npy')

binlist = []
data = []
for bat in range(3):
	for ind in indlist:
		cell = np.logical_and(batch==bat, indiv==ind)
		if np.sum(cell) > 0:
			rate_tmp = rate_bin[cell]
			read_tmp = read_bin[cell]
			s = np.sum(read_tmp>100, axis=0)
			binfilter = np.logical_and((s>=0.99*len(read_tmp)), bin_all[:,0]!='chrX')
			print(sum(cell), sum(binfilter))
			rateb = np.divide(rate_tmp[:, binfilter].T, meta[cell, 8].astype(float)).T
			readb = read_tmp[:, binfilter]
			for i in range(np.sum(binfilter)):
				rateb[readb[:,i]<100, i] = np.mean(rateb[readb[:,i]>=100, i])
			data.append(rateb)
			binlist.append(['-'.join(x.tolist()) for x in bin_all[binfilter]])

integrated, corrected, genes = scanorama.correct(data, binlist, return_dimred=True, hvg=None)
rateb_reduce = np.concatenate(integrated)
y = tsne.fit_transform(rateb_reduce[:, :ndim])

np.save(outdir + 'matrix/cell_4238_mCH_all_integrated_svd100.npy', rateb_reduce)
np.savetxt(outdir + 'matrix/cell_4238_mCH_all_integrated_svd50_p50_rs0.txt', y, delimiter='\t', fmt='%s')



