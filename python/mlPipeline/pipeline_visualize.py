def PlotSMOTE(self, X, y, X_res, y_res):
    # Two subplots, unpack the axes array immediately
    X = np.array(X)
    y = np.array(y)

    for i in range(0, (len(X[0, :]) - 1)):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        c0 = ax1.scatter(X[y == 0, i], X[y == 0, i + 1], label="Class #0 Neutral comments", alpha=0.5, color='gray')
        c1 = ax1.scatter(X[y == 1, i], X[y == 1, i + 1], label="Class #1 Emotional comments", alpha=0.5, color='red')
        ax1.set_title('Original set')

        ax2.scatter(X_res[y_res == 0, i], X_res[y_res == 0, i + 1], label="Class #0 Neutral comments", alpha=0.5,
                    color='gray')
        ax2.scatter(X_res[y_res == 1, i], X_res[y_res == 1, i + 1], label="Class #1 Emotional comments", alpha=0.5,
                    color='red')
        ax2.set_title('SMOTE')

        # make nice plotting
        for ax in (ax1, ax2):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.spines['left'].set_position(('outward', 10))
            ax.spines['bottom'].set_position(('outward', 10))
            ax.set_xlim([-6, 8])
            ax.set_ylim([-6, 6])

        fig.legend((c0, c1), ('Class #0 Neutral', 'Class #1 Emotional'), loc='lower center',
                   ncol=2, labelspacing=0.)
        plt.tight_layout(pad=3)
        fig.savefig(self.figuresOut + '/' + str(i) + '.png')
        plt.close(fig)


def PlotSMOTEPCA(self, X, y, X_res, y_res):
    # Two subplots, unpack the axes array immediately
    X = np.array(X)
    y = np.array(y)

    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X)
    X_vis_res = pca.fit_transform(X_res)

    f, (ax1, ax2) = plt.subplots(1, 2)

    c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0 Neutral comments", alpha=0.5, color='gray')
    c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1 Emotional comments", alpha=0.5, color='red')
    ax1.set_title('Original set')

    ax2.scatter(X_vis_res[y_res == 0, 0], X_vis_res[y_res == 0, 1], label="Class #0 Neutral comments", alpha=0.5,
                color='gray')
    ax2.scatter(X_vis_res[y_res == 1, 0], X_vis_res[y_res == 1, 1], label="Class #1 Emotional comments", alpha=0.5,
                color='red')
    ax2.set_title('SMOTE')

    # make nice plotting
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim([-6, 8])
        ax.set_ylim([-6, 6])

    f.legend((c0, c1), ('Class #0 Neutral', 'Class #1 Emotional'), loc='lower center',
             ncol=2, labelspacing=0.)
    plt.tight_layout(pad=3)
    plt.show()

