import torch
import torch.nn as nn
import pandas as pd
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class ODEPhysics(nn.Module):
    def __init__(self, num_trajectories, context_dim=4):
        super().__init__()
        self.context = nn.Embedding(num_trajectories, context_dim)

        self.net = nn.Sequential(
            nn.Linear(2 + context_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

        self.current_context = None

    def set_context(self, traj_ids):
        self.current_context = self.context(traj_ids)

    def forward(self, t, y):
        if self.current_context is None:
            raise RuntimeError("Context not set before ODE solve.")

        if y.dim() == 1:
            ctx = self.current_context
        else:
            ctx = self.current_context

        inp = torch.cat([y, ctx], dim=-1)
        return self.net(inp)

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    scale = 4096.0 
    trajectories = []
    for _, group in df.groupby('filename'):
        group = group.sort_values('timepoint')
        data = group[['rabbit_population', 'grass_population']].values / scale
        trajectories.append(torch.tensor(data, dtype=torch.float32))
    
    return trajectories

def train_ode(csv_path, epochs=15):
    trajectories = prepare_data(csv_path)
    model = ODEPhysics(len(trajectories))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    data_tensor = torch.stack(trajectories).to("mps")

    traj_ids = torch.arange(len(trajectories))
    dataset = TensorDataset(data_tensor, traj_ids)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    t = torch.linspace(0, 1, data_tensor.size(1), device="mps")
    
    print(f"Starting training on {len(trajectories)} trajectories...")
    
    model.to("mps")
    model.train()
    for epoch in range(epochs):
        seq_len = min(data_tensor.size(1), 2 ** (epoch // 4 + 2)) # Increase sequence length over epochs
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            traj, traj_id = batch
            model.set_context(traj_id.to("mps"))
            traj = traj[:, :seq_len, :]  # Use only the first seq_len timepoints
            y0 = traj[:, 0, :]
            t_sub = t[:seq_len]
            pred = odeint(model, y0, t_sub, method='rk4')
            pred = pred.permute(1, 0, 2)  # shape: [batch_size, seq_len, num_channels]
            pred = pred[:, :seq_len, :]
            
            loss = criterion(pred, traj)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch:03d} | Sequence length: {seq_len} | Loss: {total_loss/len(trajectories):.6f}")

    return model

def plot_results(model, trajectories, scale=4096.0):
    model.to("mps")
    model.eval()
    with torch.no_grad():
        traj = trajectories[0].to("mps")
        traj_id = torch.tensor([0], device="mps")
        model.set_context(traj_id)
        y0 = traj[0]
        y0 = y0.unsqueeze(0)
        t = torch.linspace(0, 1, len(traj), device="mps")
        pred = odeint(model, y0, t, method='rk4')
        true_counts = traj.cpu().numpy() * scale
        pred_counts = pred.cpu().numpy() * scale
        pred_counts = pred_counts.squeeze(1)  # shape: [seq_len, num_channels]
        time_axis = t.cpu().numpy()

    plt.figure(figsize=(12, 5))

    # Plot Rabbits
    plt.subplot(1, 2, 1)
    plt.plot(time_axis, true_counts[:, 0], 'o', label='Real Rabbits', alpha=0.6)
    plt.plot(time_axis, pred_counts[:, 0], '-', label='ODE Predicted', linewidth=2)
    plt.title("Rabbit Population Dynamics")
    plt.xlabel("Timepoint")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Grass
    plt.subplot(1, 2, 2)
    plt.plot(time_axis, true_counts[:, 1], 's', label='Real Grass', color='green', alpha=0.6)
    plt.plot(time_axis, pred_counts[:, 1], '-', label='ODE Predicted', color='darkgreen', linewidth=2)
    plt.title("Grass Population Dynamics")
    plt.xlabel("Timepoint")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_context_embeddings(model, trajectories, extinction_threshold=1e-3):
    """
    Plot the learned trajectory context embeddings using PCA and color
    by behavioral regime (rabbit extinction vs survival).
    """
    model.eval()
    with torch.no_grad():
        emb = model.context.weight.detach().cpu()  # [num_trajectories, context_dim]

        emb_centered = emb - emb.mean(0)
        U, S, V = torch.pca_lowrank(emb_centered, q=2)
        emb_2d = emb_centered @ V[:, :2]

        x = emb_2d[:, 0].numpy()
        y = emb_2d[:, 1].numpy()

    # Determine regime for each trajectory
    regimes = []
    for traj in trajectories:
        final_rabbits = traj[-1, 0].item()
        if final_rabbits < extinction_threshold:
            regimes.append("extinction")
        else:
            regimes.append("survival")

    colors = ["red" if r == "extinction" else "blue" for r in regimes]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=colors, alpha=0.8)

    for i in range(len(x)):
        plt.text(x[i], y[i], str(i), fontsize=8, alpha=0.7)

    plt.title("Context Embeddings Colored by Dynamics Regime")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


trajectories = prepare_data('../../../data/rabbit_grass_sweeps_oscil/rabbit_grass_sweeps_oscil_summary_stats.csv')
trained_model = train_ode('../../../data/rabbit_grass_sweeps_oscil/rabbit_grass_sweeps_oscil_summary_stats.csv')
plot_results(trained_model, trajectories)
plot_context_embeddings(trained_model, trajectories)
