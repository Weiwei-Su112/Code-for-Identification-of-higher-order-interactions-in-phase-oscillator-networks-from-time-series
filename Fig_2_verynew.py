import shutil
from GeneralInteraction import *

import matplotlib.pyplot as plt


T = 900
dt = 0.02
p = 0.1
starts_from = 1 / 9
cut_time = int(T * starts_from)

conn_seed = 1020
noise_seed = 3999
noise_seed2, noise_seed3, noise_seedmix = np.random.default_rng(noise_seed).integers(0, 1e10, size=3)
reduce_seed = 100912

plt_x_axis = np.arange(-cut_time, T - cut_time, dt)

need_sps = [100.0, 100.0, 100.0]
need_start = 0
need_end = 600
inf_last = (need_start + need_end) / T
index_len = int((need_start + need_end) / dt)

# 0510:
v_mixes = [[100.0], [100.0], [100.0]]
plt_x_axis_2 = np.arange(0, need_end + need_start, dt)[:index_len]
plt_x_axis_3 = np.arange(0, need_end + need_start, dt)[:index_len]
plt_x_axis_mix = np.arange(0, need_end + need_start, dt)[:index_len]

plt_x_axis_lst = [plt_x_axis_2, plt_x_axis_3, plt_x_axis_mix]

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']

now = datetime.now()
date = now.strftime("%Y%m%d")
tm_string = now.strftime("%H%M%S") + "_" + str(conn_seed)

# ==============================================================================
cou2 = 0.1
cou3 = 0.3
p2 = 0.06
p3 = 0.01
# ==============================================================================
# 2

some_rng = np.random.default_rng(9876)
natfreq = some_rng.normal(loc=1, scale=0.1, size=12)
new_model_2 = GeneralInteraction(coupling2=cou2, coupling3=0, dt=dt, T=T, natfreqs=natfreq, with_noise=True,
                                 noise_sth=0.2, normalize=True, conn=p2, all_connected=False,
                                 conn_seed=conn_seed, noise_seed=noise_seed2, starts_from=starts_from,
                                 inf_last=inf_last)
test_conn2 = new_model_2.conn_mat2
test_coup2 = new_model_2.coupling2
init_phase = new_model_2.init_phase

# model_2.init_ang = init_ang
act_mat_2 = new_model_2.run()
# __, _ = plot_phase_coherence(act_mat_2, color="blue", coup2=cou2, x_axis=plt_x_axis_2)
# plt.show()

# 3
new_model_3 = GeneralInteraction(coupling2=0, coupling3=cou3, dt=dt, T=T, natfreqs=natfreq, with_noise=True,
                                 noise_sth=0.2, normalize=True, conn=p3, all_connected=False, init_phase=init_phase,
                                 conn_seed=conn_seed, noise_seed=noise_seed3, starts_from=starts_from,
                                 inf_last=inf_last)

test_conn3 = new_model_3.conn_mat3
test_coup3 = new_model_3.coupling3
act_mat_3 = new_model_3.run()
# __, _ = plot_phase_coherence(act_mat_3, color="orange", coup3=cou3, x_axis=plt_x_axis_3)
# plt.show()
# natfreq = np.load(f"Fig2data/conn_data/natfreq_{str(previous_seed)}.npy")
# init_ang = np.load(f"Fig2data/conn_data/initphase_{str(previous_seed)}.npy")

# mix
test_conn_mix2 = reduce_conn_2(test_conn2, 0.5, reduce_seed)
test_conn_mix3 = reduce_conn_3(test_conn3, 0.5, reduce_seed)
test_coup_mix2 = test_conn_mix2 * cou2 / (p2 * 12)
test_coup_mix3 = test_conn_mix3 * cou3 / (p3 * 12 * 12 / 2)
new_model_mix = GeneralInteraction(dt=dt, T=T, natfreqs=natfreq, pre_coup2=test_coup_mix2, pre_coup3=test_coup_mix3,
                                   pre_conn2=test_conn_mix2, pre_conn3=test_conn_mix3, normalize=True, noise_sth=.2,
                                   noise_seed=noise_seedmix, init_phase=init_phase, starts_from=starts_from,
                                   inf_last=inf_last)
act_mat_mix = new_model_mix.run()

# __, _ = plot_phase_coherence(act_mat_mix, color="green", coup2=cou2, coup3=cou3, x_axis=plt_x_axis_mix)
# plt.show()
plot_phase_coherence_3set_all(act_mat_2, act_mat_3, act_mat_mix, x_axis=plt_x_axis_lst,
                              coup2=cou2, coup3=cou3, coup2_mix=cou2,
                              coup3_mix=cou3)
plt.show()
# # ==============================================================================
act_mat_lst = [act_mat_2, act_mat_3, act_mat_mix]
type_is_lst = ["Pairwise", "3-interaction", "Mixture"]
new_model_lst = [new_model_2, new_model_3, new_model_mix]

file_path = "Fig2data/" + date + "/"
os.makedirs(file_path, exist_ok=True)
base_path = "Fig2data/" + date + "/" + tm_string + "/"
os.mkdir(base_path)
path_2 = "Fig2data/" + date + "/" + tm_string + "/Pairwise/"
path_3 = "Fig2data/" + date + "/" + tm_string + "/3-interaction/"
path_mix = "Fig2data/" + date + "/" + tm_string + "/Mixture/"
path_lst = [path_2, "../3-interaction/", "../Mixture/"]
os.mkdir(path_2)
os.mkdir(path_3)
os.mkdir(path_mix)

for ptr in range(0, 3):
    act_mat = act_mat_lst[ptr]
    type_is = type_is_lst[ptr]
    new_model = new_model_lst[ptr]
    root_path = path_lst[ptr]

    file_name0 = root_path + "natfreqs.csv"
    file_name1 = root_path + "pairwise.csv"
    file_name2 = root_path + "3-interaction.csv"
    file_name_confusion = root_path + "confusion_values.csv"
    demo_coup2 = new_model.coupling2
    demo_coup3 = new_model.coupling3

    norma_2 = np.max(demo_coup2)
    if norma_2 == 0:
        norma_2 = 1
    norma_3 = np.max(demo_coup3)
    if norma_3 == 0:
        norma_3 = 1

    header0 = ["i", "Real", "Estimate - LASSO", "Estimate - MLE", "Estimate - Ada. LASSO"]
    header2 = ["i", "j", "Real", "Estimate - LASSO", "Estimate - MLE", "Estimate - Ada. LASSO"]
    header3 = ["i", "j", "k", "Real", "Estimate - LASSO", "Estimate - MLE", "Estimate - Ada. LASSO"]
    header_confusion = ["Confusion Values", "Estimate - Ada. LASSO", "Estimate - LASSO", "Estimate - MLE"]

    natfreq_lst = []
    lst_2 = []
    lst_3 = []

    lst_2_t_real = np.array([])
    lst_2_t_lasso = np.array([])
    lst_2_t_mle = np.array([])
    lst_2_t_lasso_ada = np.array([])
    lst_3_t_real = np.array([])
    lst_3_t_lasso = np.array([])
    lst_3_t_mle = np.array([])
    lst_3_t_lasso_ada = np.array([])

    prepare_all = new_model.prepare_diffs(act_mat)
    mle_or_ols_results = new_model.solve_ols(all_prepared=prepare_all)
    ada_results = new_model.solve_ada_lasso(all_prepared=prepare_all, mle_or_ols_results=mle_or_ols_results)
    lasso_results = new_model.solve_lasso(all_prepared=prepare_all)
    fdr_results, _ = new_model.fdr_control_for_mle(mle_or_ols_results)
    mle_coup = new_model.mle_with_fdr(mle_or_ols_results, fdr_results)

    for i in range(act_mat.shape[0]):
        others_lst = np.delete(new_model.all_nodes, i)
        more_others_lst = new_model.make_more_others_lst(others_lst, i)

        demo_lst2 = ["Real-2", new_model.natfreqs[i] % (2 * np.pi)]
        for j in others_lst:
            demo_lst2.append(demo_coup2[j][i])
        real_2 = demo_lst2[2:]

        demo_lst3 = ["Real-3"]
        for j in others_lst:
            if j == others_lst[-1]:
                break
            elif j < i:
                for k in more_others_lst[j]:
                    demo_lst3.append(demo_coup3[i][j][k])
            else:
                for k in more_others_lst[j - 1]:
                    demo_lst3.append(demo_coup3[i][j][k])
        real_3 = demo_lst3[1:]

        lasso_2 = lasso_results[i]["2"]
        mle_2 = mle_coup[i]["2"]
        lasso_2_ada = ada_results[i]["2"]
        lasso_3 = lasso_results[i]["3"]
        mle_3 = mle_coup[i]["3"]
        lasso_3_ada = ada_results[i]["3"]

        counter_2 = 0
        col_names2 = ["", "natural frequencies_" + str(i + 1)]
        for index in others_lst:
            name = "k_" + str(index + 1) + str(i + 1)
            col_names2.append(name)
            inputs = [str(i + 1), str(index + 1), real_2[counter_2] / norma_2,
                      lasso_2[counter_2] / norma_2, mle_2[counter_2] / norma_2,
                      lasso_2_ada[counter_2] / norma_2]
            lst_2.append(inputs)
            counter_2 += 1

        counter_3 = 0
        col_names3 = [""]
        for index in others_lst:
            if index == others_lst[-1]:
                break
            elif index < i:
                for inde2 in more_others_lst[index]:
                    name = "k_" + str(index + 1) + str(inde2 + 1) + str(i + 1)
                    col_names3.append(name)
                    inputs = [str(i + 1), str(index + 1), str(inde2 + 1), real_3[counter_3] / norma_3,
                              lasso_3[counter_3] / norma_3, mle_3[counter_3] / norma_3,
                              lasso_3_ada[counter_3] / norma_3]
                    lst_3.append(inputs)
                    counter_3 += 1
            else:
                for inde2 in more_others_lst[index - 1]:
                    name = "k_" + str(index + 1) + str(inde2 + 1) + str(i + 1)
                    col_names3.append(name)
                    inputs = [str(i + 1), str(index + 1), str(inde2 + 1), real_3[counter_3] / norma_3,
                              lasso_3[counter_3] / norma_3, mle_3[counter_3] / norma_3,
                              lasso_3_ada[counter_3] / norma_3]
                    lst_3.append(inputs)
                    counter_3 += 1

        natfreq_lst.append([str(i + 1), new_model.natfreqs[i] % (2 * np.pi), lasso_results[i]["natfreq"],
                            mle_or_ols_results[i]["natfreq"], ada_results[i]["natfreq"]])

    with open(file_name_confusion, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_confusion)
        all_res = new_model.conn_criteria_base(mle_coup, lasso_results, ada_results)

        ada_conn = all_res["ada"]
        lasso_conn = all_res["lasso"]
        mle_conn = all_res["mle"]

        writer.writerow(["TP_2", ada_conn["TP_2"], lasso_conn["TP_2"], mle_conn["TP_2"]])
        writer.writerow(["TN_2", ada_conn["TN_2"], lasso_conn["TN_2"], mle_conn["TN_2"]])
        writer.writerow(["FP_2", ada_conn["FP_2"], lasso_conn["FP_2"], mle_conn["FP_2"]])
        writer.writerow(["FN_2", ada_conn["FN_2"], lasso_conn["FN_2"], mle_conn["FN_2"]])
        writer.writerow(["TP_3", ada_conn["TP_3"], lasso_conn["TP_3"], mle_conn["TP_3"]])
        writer.writerow(["TN_3", ada_conn["TN_3"], lasso_conn["TN_3"], mle_conn["TN_3"]])
        writer.writerow(["FP_3", ada_conn["FP_3"], lasso_conn["FP_3"], mle_conn["FP_3"]])
        writer.writerow(["FN_3", ada_conn["FN_3"], lasso_conn["FN_3"], mle_conn["FN_3"]])
        f.close()

    with open(file_name0, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header0)
        writer.writerows(natfreq_lst)
        # somehow for-loop every row with the information
        f.close()

    with open(file_name1, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header2)
        writer.writerows(lst_2)
        # somehow for-loop every row with the information
        f.close()

    with open(file_name2, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header3)
        writer.writerows(lst_3)
        # somehow for-loop every row with the information
        f.close()

    # output_zip_name = date + tm_string + type_is
    os.chdir(root_path)
    # shutil.make_archive(output_zip_name, 'zip')
