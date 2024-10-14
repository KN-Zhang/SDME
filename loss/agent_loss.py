import torch
import torch.nn as nn

class AgentLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'agent'
    def forward(self, updated_agents, **kw):
        loss = 0.
        for agent in updated_agents:
            b, n, c = agent.shape
            tmp = agent.unsqueeze(-2).repeat(1, 1, n, 1)  ## b n n c             
            tmp_T = tmp.permute(0, 2, 1, 3)
            tmp2 = torch.sum(tmp * tmp_T, dim=-1) / ( torch.norm(tmp, dim=-1) * torch.norm(tmp_T, dim=-1) + 1e-5 )
            tmp2 = tmp2 - torch.eye(n)[None, :, :].to(tmp2) * tmp2
            loss = loss + torch.mean(torch.sum(tmp2.reshape(b, -1), dim=-1) / (n * (n-1)))
        return loss