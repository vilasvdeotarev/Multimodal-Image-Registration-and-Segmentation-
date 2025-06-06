import numpy as np
import cv2 as cv
import warnings
from matplotlib import pylab
from prettytable import PrettyTable
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)

    Algorithm = ['Terms', 'ZOA-ATRSNet', 'WOA-ATRSNet', 'KOA-ATRSNet', 'LEOA-ATRSNet', 'HK-LEOA-ATRSNet']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((Fitness.shape[-2], 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = stats(Fitness[j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Report ',
          ' Dataset --------------------------------------------------')

    print(Table)
    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness
    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, markersize=12, label=Algorithm[1])
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, markersize=12, label=Algorithm[2])
    plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, markersize=12, label=Algorithm[3])
    plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, markersize=12, label=Algorithm[4])
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, markersize=12, label=Algorithm[5])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Convergence.png")
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.show()


def plot_results_Seg():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']

    Full = ['TERMS', 'ZOA-ATRSNet', 'WOA-ATRSNet', 'KOA-ATRSNet', 'LEOA-ATRSNet', 'HK-LEOA-ATRSNet',
            'ANN', 'Unet', 'ResUnet', 'ATRSNet', 'HK-LEOA-ATRSNet']
    value_all = Eval_all
    stats = np.zeros((Eval_all.shape[-1] - 4, Eval_all.shape[0] + 4, 5))  # (METRICS, ALGORITHM, STATS)
    for i in range(4, Eval_all.shape[-1] - 6):
        for j in range(Eval_all.shape[0] + 4):
            if j < value_all.shape[0]:
                stats[i, j, 0] = np.max(value_all[j][:, i])
                stats[i, j, 1] = np.min(value_all[j][:, i])
                stats[i, j, 2] = np.mean(value_all[j][:, i])
                stats[i, j, 3] = np.median(value_all[j][:, i])
                stats[i, j, 4] = np.std(value_all[j][:, i])

        if i == 7:
            pass
        else:
            temp = stats[i, 4, :]
            stats[i, 9, :] = temp

            # Table = PrettyTable()
            # Table.add_column(Full[0], Statistics)
            # for k in range(len(Full) - 1):
            #     Table.add_column(Full[k + 1], stats[i, k, :])
            # print('--------------------------------------------------Segmentation Comparison',
            #       Terms[i - 4],
            #       '--------------------------------------------------')
            # print(Table)

            Alg_Value = stats[i, :5, :]
            barWidth = 0.15
            # colors = ['#e63946', '#457b9d', '#000000', '#c04e01', '#be03fd']
            colors = ['#ceb301', '#677a04', '#ffb07c', '#cea2fd', '#0504aa']
            fig, ax = plt.subplots(figsize=(10, 6))
            X = np.arange(len(Statistics))
            y_min = np.min(Alg_Value)
            y_max = np.max(Alg_Value)
            y_range = y_max - y_min
            if y_range < 1:
                y_min -= 0.05
                y_max += 0.05
            else:
                y_min -= y_range * 0.05
                y_max += y_range * 0.1
            ax.set_ylim([0, y_max])
            for l in range(X.shape[-1]):
                bars = ax.bar(X + l * barWidth * 1.15, Alg_Value[l, :], color=colors[l], width=barWidth,
                              edgecolor='#032b43',
                              label=Full[l + 1])
                for b, bar in enumerate(bars):
                    height = bar.get_height()
                    text_offset = y_range * 0.012  # 0.02
                    algo_offset = y_range * 0.05  # 0.05
                    # if (i == 0 and (l == 0 or l == 4)) or (i == 1 and l == 1) or (i == 2 and l == 2) or (i == 3 and l == 3):
                    if ((b == 0 and l == 0) or (b == 1 and l == 1) or (b == 2 and l == 2) or (b == 3 and l == 3)
                            or (b == 4 and l == 4)):
                        ax.text(bar.get_x() + bar.get_width() / 2, height + text_offset, f"{height:.2f}", ha='center',
                                va='bottom',
                                fontsize=8, color="k",
                                bbox=dict(edgecolor='k', boxstyle='larrow ,pad=0.2', fc="k", alpha=0.9), rotation=90)

                        ax.text(bar.get_x() + bar.get_width() / 2, height + algo_offset, Full[l + 1], ha='center',
                                va='bottom',
                                fontsize=10, color="w",
                                bbox=dict(edgecolor='k', boxstyle='square,pad=0.3', fc="k", alpha=0.9), rotation=0)

            ax.set_xlabel('Statisticsal Analysis', fontsize=12, fontweight='bold')
            ax.set_ylabel(Terms[i - 4], fontsize=12, fontweight='bold')
            ax.set_xticks(X + (((len(Full[:6]) - 1) * barWidth) / 2.2))
            ax.set_xticklabels(Statistics)
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=False, shadow=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)
            ax.grid(which='major', axis='y', linestyle='-')

            path = "./Results/Segmentation_%s_Alg.png" % (Terms[i - 4])
            plt.savefig(path)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Statisticsal Analysis vs ' + Terms[i - 4] + 'Algorithm Comparision')
            plt.show()

            # method Datas
            Mtd_Value = stats[i, 5:, :]
            barWidth = 0.15
            # colors = ['#e63946', '#457b9d', '#000000', '#c04e01', '#be03fd]
            colors = ['#c0fb2d', '#13eac9', '#03719c', '#f10c45', '#fdaa48']
            fig, ax = plt.subplots(figsize=(10, 6))
            X = np.arange(len(Statistics))
            y_min = np.min(Mtd_Value)
            y_max = np.max(Mtd_Value)
            y_range = y_max - y_min
            if y_range < 1:
                y_min -= 0.05
                y_max += 0.05
            else:
                y_min -= y_range * 0.05
                y_max += y_range * 0.1
            ax.set_ylim([0, y_max])
            for l in range(X.shape[-1]):
                bars = ax.bar(X + l * barWidth * 1.15, Mtd_Value[l, :], color=colors[l], width=barWidth,
                              edgecolor='#032b43',
                              label=Full[5:][l + 1])
                for b, bar in enumerate(bars):
                    height = bar.get_height()
                    text_offset = y_range * 0.012
                    algo_offset = y_range * 0.05
                    # if (i == 0 and (l == 0 or l == 4)) or (i == 1 and l == 1) or (i == 2 and l == 2) or (i == 3 and l == 3):
                    if ((b == 0 and l == 0) or (b == 1 and l == 1) or (b == 2 and l == 2) or (b == 3 and l == 3)
                            or (b == 4 and l == 4)):
                        ax.text(bar.get_x() + bar.get_width() / 2, height + text_offset, f"{height:.2f}", ha='center',
                                va='bottom',
                                fontsize=8, color="k",
                                bbox=dict(edgecolor='k', boxstyle='larrow ,pad=0.2', fc="k", alpha=0.9), rotation=90)

                        ax.text(bar.get_x() + bar.get_width() / 2, height + algo_offset, Full[5:][l+1], ha='center',
                                va='bottom',
                                fontsize=10, color="w",
                                bbox=dict(edgecolor='k', boxstyle='square,pad=0.3', fc="k", alpha=0.9), rotation=0)

            ax.set_xlabel('Statisticsal Analysis', fontsize=12, fontweight='bold')
            ax.set_ylabel(Terms[i - 4], fontsize=12, fontweight='bold')
            ax.set_xticks(X + (((len(Full[:6]) - 1) * barWidth) / 2.2))
            ax.set_xticklabels(Statistics)
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=False, shadow=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)
            ax.grid(which='major', axis='y', linestyle='-')
            path = "./Results/Segmentation_%s_MTD.png" % (Terms[i - 4])
            plt.savefig(path)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Statisticsal Analysis vs ' + Terms[i - 4] + 'Method Comparision')
            plt.show()


def plot_Images_vs_terms():
    eval = np.load('Eval_all_img.npy', allow_pickle=True)
    Terms = ['SSIM', 'Mutual Information', 'RMSE', 'Correlation Coefficient']
    Graph_Terms = [0, 1, 2]
    table_Terms = [3]

    Algorithm = ['Terms', 'ZOA-ATRSNet', 'WOA-ATRSNet', 'KOA-ATRSNet', 'LEOA-ATRSNet', 'HK-LEOA-ATRSNet']
    Classifier = ['Terms', 'ANN', 'Unet', 'ResUnet', 'ATRSNet', 'HK-LEOA-ATRSNet']

    Images = ['1', '2', '3', '4', '5']
    value = eval[4, :, :]
    Table = PrettyTable()
    Table.add_column(Algorithm[0], np.asarray(Terms)[table_Terms])
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value[j, table_Terms])
    print('-------------------------------------------------- Images Algorithm Comparison',
          '--------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], np.asarray(Terms)[table_Terms])
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, table_Terms])
    print('-------------------------------------------------- Images Classifier Comparison',
          '--------------------------------------------------')
    print(Table)

    for j in range(len(Graph_Terms)):
        Graph = np.zeros((eval.shape[0], eval.shape[1] + 1))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Terms[j]]

        # Algorithm Comparisions
        Alg_Value = Graph[:, :5]
        barWidth = 0.15
        # colors = ['#e63946', '#457b9d', '#000000', '#c04e01', '#be03fd']
        colors = ['#ceb301', '#677a04', '#ffb07c', '#cea2fd', '#0504aa']
        fig, ax = plt.subplots(figsize=(10, 6))
        X = np.arange(len(Images))
        y_min = np.min(Graph)
        y_max = np.max(Graph)
        y_range = y_max - y_min
        if y_range < 1:
            y_min -= 0.05
            y_max += 0.05
        else:
            y_min -= y_range * 0.05
            y_max += y_range * 0.1
        ax.set_ylim([0, y_max])
        for l in range(X.shape[-1]):
            bars = ax.bar(X + l * barWidth * 1.15, Alg_Value[:, l], color=colors[l], width=barWidth, edgecolor=colors[l],  # '#032b43'
                          label=Algorithm[l+1])
            for i, bar in enumerate(bars):
                height = bar.get_height()
                text_offset = y_range * 0.012
                algo_offset = y_range * 0.05
                # if (i == 0 and (l == 0 or l == 4)) or (i == 1 and l == 1) or (i == 2 and l == 2) or (i == 3 and l == 3):
                if ((i == 0 and l == 0) or (i == 1 and l == 1) or (i == 2 and l == 2) or (i == 3 and l == 3)
                        or (i == 4 and l == 4)):
                    ax.text(bar.get_x() + bar.get_width() / 2, height + text_offset, f"{height:.2f}", ha='center',
                            va='bottom',
                            fontsize=8, color="k",
                            bbox=dict(edgecolor='k', boxstyle='larrow ,pad=0.2', fc="k", alpha=0.9), rotation=90)

                    ax.text(bar.get_x() + bar.get_width() / 2, height + algo_offset, Algorithm[l+1], ha='center',
                            va='bottom',
                            fontsize=10, color="w",
                            bbox=dict(edgecolor='k', boxstyle='square,pad=0.3', fc="k", alpha=0.9), rotation=0)
        ax.set_xlabel('Images', fontsize=12, fontweight='bold')
        ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold')
        ax.set_xticks(X + (((len(Algorithm)-1) * barWidth) / 2.2))
        ax.set_xticklabels(Images)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=False, shadow=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)
        ax.grid(which='major', axis='y', linestyle='-')
        path = "./Results/%s_bar_alg.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Images vs ' + Terms[Graph_Terms[j]] + ' Algorithm comparision')
        plt.savefig(path)
        plt.show()

        # Method Comparisions
        Mtd_Value = Graph[:, 5:]
        barWidth = 0.15
        # colors = ['#e63946', '#457b9d', '#000000', '#c04e01', '#be03fd']
        colors = ['#c0fb2d', '#13eac9', '#03719c', '#f10c45', '#fdaa48']
        fig, ax = plt.subplots(figsize=(10, 6))
        X = np.arange(len(Images))
        y_min = np.min(Graph)
        y_max = np.max(Graph)
        y_range = y_max - y_min
        if y_range < 1:
            y_min -= 0.05
            y_max += 0.05
        else:
            y_min -= y_range * 0.05
            y_max += y_range * 0.1
        ax.set_ylim([0, y_max])
        for l in range(X.shape[-1]):
            bars = ax.bar(X + l * barWidth * 1.15, Mtd_Value[:, l], color=colors[l], width=barWidth, edgecolor=colors[l],
                          label=Classifier[l + 1])
            for i, bar in enumerate(bars):
                height = bar.get_height()
                text_offset = y_range * 0.012
                algo_offset = y_range * 0.05
                if ((i == 0 and l == 0) or (i == 1 and l == 1) or (i == 2 and l == 2) or (i == 3 and l == 3)
                        or (i == 4 and l == 4)):
                    ax.text(bar.get_x() + bar.get_width() / 2, height + text_offset, f"{height:.2f}", ha='center',
                            va='bottom',
                            fontsize=8, color="k",
                            bbox=dict(edgecolor='k', boxstyle='larrow ,pad=0.2', fc="k", alpha=0.9), rotation=90)

                    ax.text(bar.get_x() + bar.get_width() / 2, height + algo_offset, Classifier[l + 1], ha='center',
                            va='bottom',
                            fontsize=10, color="w",
                            bbox=dict(edgecolor='k', boxstyle='square,pad=0.3', fc="k", alpha=0.9), rotation=0)

        ax.set_xlabel('Images', fontsize=12, fontweight='bold')
        ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold')
        ax.set_xticks(X + (((len(Classifier) - 1) * barWidth) / 2.2))
        ax.set_xticklabels(Images)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=False, shadow=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)
        ax.grid(which='major', axis='y', linestyle='-')
        path = "./Results/%s_bar_mtd.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Images vs ' + Terms[Graph_Terms[j]] + ' Method comparision')
        plt.savefig(path)
        plt.show()


def Image_Results():
    CT = np.load('CT_Images.npy')
    MR = np.load('MR_Images.npy')
    Reg = np.load('Fused_Images.npy')
    Seg = np.load('Ground_Truth.npy')
    # Image = [174, 175, 176, 185, 241, 243, 247, 340]
    Image = [175, 176, 185, 241, 243, 247]
    for i in range(5):
        print(i, len(Image))
        ct = CT[Image[i]]
        mri = MR[Image[i]]
        reg = Reg[Image[i]]
        seg = Seg[Image[i]]

        # cv.imshow('Registered image', np.uint8(reg))
        # cv.imshow("MRI image", np.uint8(mri))
        # cv.imshow("CT image", np.uint8(ct))
        # cv.imshow("Segmented image", np.uint8(seg))
        # cv.waitKey(0)

        plt.suptitle('Segmented Images from Dataset', fontsize=20)

        plt.subplot(2, 2, 1).axis('off')
        plt.imshow(ct)
        plt.title('CT', fontsize=10)

        plt.subplot(2, 2, 2).axis('off')
        plt.imshow(mri)
        plt.title('MR', fontsize=10)

        plt.subplot(2, 2, 3).axis('off')
        plt.imshow(reg)
        plt.title('Registered', fontsize=10)

        plt.subplot(2, 2, 4).axis('off')
        plt.imshow(seg)
        plt.title('Segmented', fontsize=10)
        path = "./Results/Image_Results/Compared_image_%s.png" % (i + 1)
        plt.savefig(path)
        plt.show()
        cv.imwrite('./Results/Image_Results/CT_image_' + str(i + 1) + '.png', np.uint8(ct))
        cv.imwrite('./Results/Image_Results/MR_image_' + str(i + 1) + '.png', np.uint8(mri))
        cv.imwrite('./Results/Image_Results/Registered_image_' + str(i + 1) + '.png', np.uint8(reg))
        cv.imwrite('./Results/Image_Results/segmented_image_' + str(i + 1) + '.png', np.uint8(seg))


def Sample_images():
    CT = np.load('CT_Images.npy', allow_pickle=True)
    MR = np.load('MR_Images.npy', allow_pickle=True)
    # Reg = np.load('Registered_image.npy', allow_pickle=True)
    # Seg = np.load('segmented_image.npy', allow_pickle=True)
    Images = [50, 500, 5000, 5500, 5555]
    for i in range(len(Images)):
        ct = CT[Images[i]]
        mri = MR[Images[i]]
        # reg = Reg[i]
        # seg = Seg[i]

        cv.imshow("mri image", mri)
        cv.imshow("ct image", ct)
        # cv.imshow('Registered image', np.uint8(reg))
        # cv.imshow("Segmented image", np.uint8(seg))
        cv.waitKey(750)

        cv.imwrite('./Results/Sample_Images/CT_image_' + str(i+1) + '.png', ct)
        cv.imwrite('./Results/Sample_Images/MRI_image_' + str(i+1) + '.png', mri)
        # cv.imwrite('./Results/Sample_Images/Registered_image_' + str(i - 4) + '.png', np.uint8(reg))
        # cv.imwrite('./Results/Sample_Images/segmented_image_' + str(i - 4) + '.png', np.uint8(seg))


if __name__ == '__main__':
    plot_conv()
    plot_results_Seg()
    plot_Images_vs_terms()
    Image_Results()
    Sample_images()
