resnettssd = tf.keras.Model(
            resnettssd.input, resnettssd.layers[-4].output
        )
#resnettssd.summary()

enc_results = resnettssd(x_all_tsne)
enc_results = np.array(enc_results)
X_embedded = TSNE(n_components=2).fit_transform(enc_results)
fig4 = plt.figure(figsize=(18,12))
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_all_tsne)
plt.savefig('graphs/latentspace_'+str(num_classes)+'.png')
plt.close(fig4)
#plt.show()