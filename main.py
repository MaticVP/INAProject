import utils
from dynetx import DynGraph
from networkx.algorithms.community import label_propagation_communities

def main():
    nodes, edges = utils.load_network_data("./Data/metadata_LyonSchool.dat"
                                           ,"./Data/tij_LyonSchool.dat")

    G_dyn = DynGraph()
    start_from_sample_t = 1
    end_at_sample_t = 5

    for row in edges.itertuples(index=False):
        t, u, v = row

        if (31220 + 20 *end_at_sample_t) >= t >= (31220+20 * start_from_sample_t):
            G_dyn.add_interaction(str(u), str(v), int(t))

    community_snapshots = utils.detect_communities(G_dyn, label_propagation_communities)

    utils.plot_community_sizes_over_time(community_snapshots)

    community_df = utils.community_snapshots_to_long_df(community_snapshots)

    utils.create_sankey(community_df)

    utils.plot_community_timeline(community_df, save_path="community_timeline.png")

    utils.plot_snapshots(G_dyn,community_df,layout="spring")

if __name__ == "__main__":
    main()