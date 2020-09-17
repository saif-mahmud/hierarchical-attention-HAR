import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# generate and save the required files and load
opp_window_attn = np.load('../data/attn-maps/opp_benm_window.npy')
opp_session_attn = np.load('../data/attn-maps/opp_benm_session.npy')
opp_labels = np.load('../data/attn-maps/opp_benm_labels.npy')
opp_preds = np.load('../data/attn-maps/opp_benm_preds.npy')
opp_mid_level = np.load('../data/attn-maps/opp_benm_mid_l.npy')
opp_locomotion = np.load('../data/attn-maps/opp_benm_loco_l.npy')


def show_mex_attention(idx):
    print(idx)
    print(f'True Label: {activity_map[mex_labels[idx]]}')
    print(f'Predicted: {activity_map[mex_preds[idx]+1]}')

    f,[ax_window, ax_heat]=plt.subplots(nrows=2,figsize=(10, 5.5),sharex=True, gridspec_kw={'height_ratios': [1., .5]})

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.heatmap(pd.DataFrame(mex_session_attn[idx], columns=range(1,21,2)), cmap=cmap, square=True, cbar=False, ax=ax_heat)
    time_axis = np.around(np.linspace(0,20,2000),1)

    data = pd.DataFrame(mex_data_non_scaled[idx])
    data.columns = ['thigh_x', 'thigh_y', 'thigh_z','wrist_x', 'wrist_y', 'wrist_z']
    data['thigh_magnitude'] = np.sqrt(np.square(data['thigh_x']) + np.square(data['thigh_y']) + np.square(data['thigh_z']))
    data['wrist_magnitude'] = np.sqrt(np.square(data['wrist_x']) + np.square(data['wrist_y']) + np.square(data['wrist_z']))

    ax_window.plot(time_axis, data['thigh_magnitude'],'b:', time_axis, data['wrist_magnitude'],'y:', lw=1.5)
    ax_window.legend(data.columns[-2:])


    thigh_attn = mex_window_attn[idx, 0::2, :].flatten()
    wrist_attn = mex_window_attn[idx, 1::2, :].flatten()

    thigh_sensor_masked =  np.ma.masked_where(thigh_attn < wrist_attn, data['thigh_magnitude'])
    wrist_sensor_masked =  np.ma.masked_where(wrist_attn < thigh_attn, data['wrist_magnitude'])
 
    ax_window.plot(time_axis, thigh_sensor_masked, 'b-', lw=2.0)
    ax_window.plot(time_axis, wrist_sensor_masked, 'y-', lw=2.0)
    plt.savefig('mex_attnmap-superman.jpg', dpi=200, quality=95, bbox_inches = 'tight', pad_inches = 0)
    plt.show()


def opp_plot_attnmap_as_subplots(idx):
    plt.rcParams.update({'font.size': 12})

    f,[ax_loco, ax_mid, ax_window, ax_heat]=plt.subplots(nrows=4,figsize=(10, 11.5),sharex=True, gridspec_kw={'height_ratios': [.6, .8, 1.2, .5]})

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.heatmap(pd.DataFrame(opp_session_attn[idx],
                             columns=np.around(np.linspace(0,30,36), 1)),
                 cmap=cmap, square=True, cbar=False,
                robust=False, ax=ax_heat)
    ax_heat.xaxis.set_tick_params(labelbottom=True)
    ax_heat.get_yaxis().set_visible(False)

    attn_df = pd.DataFrame({'mid_label':opp_mid_level[idx],
                    'loco_label':opp_locomotion[idx],
                    'time':np.linspace(0,36, num=25*36)})
    prev_midlabel = None
    prev_locolabel = None
    mid_line_sequence = []
    loco_line_sequence = []
    mid_label_list = []
    loco_label_list = []
    for i, row in attn_df.iterrows():
        current_midlabel = row['mid_label']
        if current_midlabel != prev_midlabel:
            prev_midlabel = current_midlabel
            mid_line_sequence.append(row['time'])
            mid_label_list.append(current_midlabel)

        current_locolabel = row['loco_label']
        if current_locolabel != prev_locolabel:
            prev_locolabel = current_locolabel
            loco_line_sequence.append(row['time'])
            loco_label_list.append(current_locolabel)
    loco_line_sequence.append(36.0)
    loco_label_list.append(prev_locolabel)
    mid_line_sequence.append(36.0)
    mid_label_list.append(prev_midlabel)

    mid_level_colors = {0:'red', 1:'limegreen', 2:'lime',3:'lightcoral',4:'indianred',5:'slateblue',6:'darkslateblue',7:'orange',8:'darkorange', 9:'teal',10:'cyan',11:'hotpink',12:'fuchsia', 13:'firebrick', 14:'maroon', 15:'royalblue', 16:'navy', 17:'purple'}
    locomotion_colors = {0:'red', 1:'yellow', 2:'green', 3:'blue', 4:'purple'}

    for i in range(len(loco_line_sequence)-1):
        color = locomotion_colors[int(loco_label_list[i])]
        ax_loco.hlines(0.1, loco_line_sequence[i], loco_line_sequence[i+1], colors=color, lw=5)
        ax_loco.text((loco_line_sequence[i]+loco_line_sequence[i+1])/2, 0.11, locomotion_activity_opp[int(loco_label_list[i])], ha='center', rotation=90)
    ax_loco.set_ylim(bottom=0.09, top=0.15)
    ax_loco.set_xlabel('Locomotion')
    ax_loco.axis('off')
    
    for i in range(len(mid_line_sequence)-1):
        color = mid_level_colors[int(mid_label_list[i])]
        ax_mid.hlines(0.1, mid_line_sequence[i], mid_line_sequence[i+1], colors=color, lw=5)
        ax_mid.text((mid_line_sequence[i] + mid_line_sequence[i+1])/2, 0.11, mid_activity_opp[int(mid_label_list[i])], ha='center', rotation=90)
    ax_mid.set_ylim(bottom=0.09, top=0.17)
    ax_mid.set_xlabel('Mid Level Gesture')
    ax_mid.axis('off')
    for label in plt.gca().xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    w_attn_df = pd.DataFrame({
                        'BACK':opp_window_attn[idx, 0::7, :].flatten(),
                        'RUA':opp_window_attn[idx, 1::7, :].flatten(),
                        'RLA':opp_window_attn[idx, 2::7, :].flatten(),
                        'LUA':opp_window_attn[idx, 3::7, :].flatten(),
                        'LLA':opp_window_attn[idx, 4::7, :].flatten(),
                        'L-SHOE':opp_window_attn[idx, 5::7, :].flatten(),
                        'R-SHOE':opp_window_attn[idx, 6::7, :].flatten()},
                        index=np.around(np.linspace(0,30,900),1)
                        )
    sns.heatmap(w_attn_df.rolling(25).mean().iloc[25-1::25].T, square=False, cmap=sns.cubehelix_palette(rot=-.3), cbar=False, ax=ax_window)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'result/attn/opp_attnmap-{idx}_{activity_map_opp[opp_labels[idx]]}_{activity_map_opp[opp_preds[idx]+1]}.jpg', dpi=200, quality=95, bbox_inches = 'tight', pad_inches = 0)
    # plt.close(f)
    # plt.ioff()
    plt.show()