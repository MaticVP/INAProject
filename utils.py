import os
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from tqdm import tqdm
import pandas as pd


def load_network_data(path_nodes, path_edges, sep="\t",
                      names_nodes=["node", "label"], names_edges=['time', 'u', 'v']):
    nodes = pd.read_csv(path_nodes, sep=sep, names=names_nodes)
    edges = pd.read_csv(path_edges, sep=sep, names=names_edges)

    return nodes, edges

class Community:
    def __init__(self,G, nodes=None, community_id=None, timestamp=None):
        self.snapshots = [G.subgraph(nodes)]
        self.community_id = community_id
        self.timestamps = [timestamp]
        self.merged_from = set()
        self.split_into = set()

    def size(self,snapshotIndex):
        return len(self.snapshots[snapshotIndex].nodes())

    def jaccard_index(self, other):
        this_prev_snapshot_index = len(self.snapshots) - 1
        other_prev_snapshot_index = len(other.snapshots) - 1

        this_prev_snapshot_nodes = set(self.snapshots[this_prev_snapshot_index].nodes())
        other_prev_snapshot_nodes = set(other.snapshots[other_prev_snapshot_index].nodes())

        return len(this_prev_snapshot_nodes & other_prev_snapshot_nodes) / len(this_prev_snapshot_nodes | other_prev_snapshot_nodes) if this_prev_snapshot_nodes | other_prev_snapshot_nodes else 0

    def __repr__(self):
        return f"{self.snapshots[len(self.snapshots)-1].nodes()}"

def detect_communities(G, algorithm, threshold_same = 0.75):
    community_snapshots = {}
    old_t = 0
    existing_communities = []
    id = 0
    for t in tqdm(sorted(G.temporal_snapshots_ids())):
        G_t = G.time_slice(t_from=old_t, t_to=t)
        if G_t.number_of_edges() == 0:
            continue
        try:

            communities = list(algorithm(G_t))
            temporal_communities = []
            communities_to_be_added = []

            for community_nodes in communities:
                new_comm = Community(G_t, nodes=community_nodes, community_id=id, timestamp=t)
                temporal_communities.append(new_comm)
                id += 1

            prev_to_curr_matches = {prev: [] for prev in existing_communities}
            curr_to_prev_matches = {curr: [] for curr in temporal_communities}

            for curr in temporal_communities:
                for prev in existing_communities:
                    jaccard = curr.jaccard_index(prev)
                    if jaccard > threshold_same:
                        curr_to_prev_matches[curr].append(prev)
                        prev_to_curr_matches[prev].append(curr)

            for curr in temporal_communities:
                matches = curr_to_prev_matches[curr]
                if len(matches) == 1:
                    prev = matches[0]
                    prev.snapshots.append(curr.snapshots[0])
                    prev.timestamps.append(t)
                    id -= 1
                elif len(matches) > 1:
                    for prev in matches:
                        prev.split_into.add(curr)
                        curr.merged_from.add(prev)
                        if prev in existing_communities:
                            existing_communities.remove(prev)
                    communities_to_be_added.append(curr)
                else:
                    communities_to_be_added.append(curr)

            for prev, matches in prev_to_curr_matches.items():
                if len(matches) > 1:
                    for curr in matches:
                        prev.split_into.add(curr)
                        curr.merged_from.add(prev)

                    if prev in existing_communities:
                        existing_communities.remove(prev)

            for comm in communities_to_be_added:
                existing_communities.append(comm)

            community_snapshots[t] = existing_communities.copy()

        except Exception as e:
            print(f"Community detection failed at time {t}: {e}")

        old_t=t
    return community_snapshots
def plot_community_sizes_over_time(community_snapshots, path="community_size_overtime.png"):
    times = sorted(community_snapshots.keys())
    all_communities = {}
    for t in times:
        for comm in community_snapshots[t]:
            if comm.community_id not in all_communities:
                all_communities[comm.community_id] = {'times': [], 'sizes': []}
            all_communities[comm.community_id]['times'].append(t)
            all_communities[comm.community_id]['sizes'].append(comm.size(-1))

    cmap = plt.get_cmap("tab20")
    num_communities = len(all_communities)
    colors = [cmap(i / num_communities) for i in range(num_communities)]

    plt.figure(figsize=(12, 6))
    for i, (comm_id, data) in enumerate(all_communities.items()):
        plt.plot(data['times'], data['sizes'], label=f'Community {comm_id}', color=colors[i])
        plt.scatter(data['times'], data['sizes'], color=colors[i], s=30)

    plt.xlabel("Time")
    plt.ylabel("Community Size")
    plt.title("Community Sizes Over Time")
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

    print(f"Saved community size over time to: {path}")

def community_snapshots_to_long_df(community_snapshots):
    records = []

    for t, communities in community_snapshots.items():
        for comm in communities:
            if t not in comm.timestamps:
                continue

            snapshot_index = comm.timestamps.index(t)
            nodes = comm.snapshots[snapshot_index].nodes()

            for node in nodes:
                records.append({
                    "node": node,
                    "community": comm.community_id,
                    "time": t
                })
    return pd.DataFrame(records)

def plot_community_timeline(df, save_path="community_timeline.png"):
    df = df.sort_values(by=["node", "time"])

    node_to_y = {node: idx for idx, node in enumerate(sorted(df["node"].unique()))}
    df["y"] = df["node"].map(node_to_y)

    communities = df["community"].unique()
    palette = sns.color_palette("hls", len(communities))
    color_map = {comm: palette[i] for i, comm in enumerate(sorted(communities))}

    plt.figure(figsize=(12, max(6, len(node_to_y) * 0.25)))
    for _, row in df.iterrows():
        plt.plot(row["time"], row["y"], 'o', color=color_map[row["community"]], markersize=5)

    plt.yticks(list(node_to_y.values()), list(node_to_y.keys()), fontsize=8)
    plt.xlabel("Time")
    plt.ylabel("Node")
    plt.title("Node Community Membership Over Time")

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"Community {comm}",
                          markerfacecolor=color_map[comm], markersize=6)
               for comm in sorted(communities)]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved community timeline to: {save_path}")

def plot_snapshots(dyn_graph, df, layout="spring", output_folder="snapshots"):
    os.makedirs(output_folder, exist_ok=True)

    times = sorted(dyn_graph.temporal_snapshots_ids())
    if not times:
        print("No snapshots to save.")
        return

    G_union = nx.Graph()
    for t in times:
        G_t = dyn_graph.time_slice(t_from=t, t_to=t + 1)
        G_union.add_nodes_from(G_t.nodes())
        G_union.add_edges_from(G_t.edges())

    if layout == "spring":
        pos = nx.spring_layout(G_union, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G_union)
    elif layout == "circular":
        pos = nx.circular_layout(G_union)
    else:
        pos = nx.spring_layout(G_union, seed=42)

    all_communities = sorted(df["community"].unique())
    palette = sns.color_palette("hls", len(all_communities))
    comm_color_map = {comm: palette[i] for i, comm in enumerate(all_communities)}

    for t in times:
        G_t = dyn_graph.time_slice(t_from=t, t_to=t + 1)
        df_t = df[df["time"] == t].set_index("node")

        node_colors = [
            comm_color_map[df_t.loc[n, "community"]] if n in df_t.index else (0.8, 0.8, 0.8)
            for n in G_t.nodes()
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"Snapshot at t = {t}")
        ax.axis('off')

        nx.draw_networkx(
            G_t,
            pos=pos,
            ax=ax,
            node_color=node_colors,
            labels={n: str(n) for n in G_t.nodes()},
            font_size=8,
            node_size=300,
            edge_color='gray',
        )

        filename = os.path.join(output_folder, f"snapshot_t{t}.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)

    print(f"Saved {len(times)} labeled snapshots to '{output_folder}/'")

def create_sankey(df, filename="community_evolution_sankey.html"):
    import plotly.graph_objects as go

    timestamps = sorted(df["time"].unique())
    label_tuples = []
    for t in timestamps:
        comms = df[df["time"] == t]["community"].unique()
        for c in comms:
            label_tuples.append((t, c))

    label_to_index = {label: idx for idx, label in enumerate(label_tuples)}
    labels = [f"t={t}, c={c}" for t, c in label_tuples]

    source = []
    target = []
    value = []

    for i in range(len(timestamps) - 1):
        t1, t2 = timestamps[i], timestamps[i + 1]
        df1 = df[df["time"] == t1]
        df2 = df[df["time"] == t2]

        merged = pd.merge(df1, df2, on="node", suffixes=("_1", "_2"))
        grouped = merged.groupby(["community_1", "community_2"]).size().reset_index(name="count")

        for _, row in grouped.iterrows():
            src = label_to_index[(t1, row["community_1"])]
            tgt = label_to_index[(t2, row["community_2"])]
            source.append(src)
            target.append(tgt)
            value.append(row["count"])

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
        ))])

    fig.update_layout(title_text="Community Evolution Sankey Diagram", font_size=10)

    if filename.endswith(".html"):
        fig.write_html(filename)
    else:
        fig.write_image(filename)

    print(f"Sankey diagram saved to: {filename}")
