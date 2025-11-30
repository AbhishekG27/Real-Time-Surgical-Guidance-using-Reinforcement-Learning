from .ddpg import DDPG
from .ddpgbc import DDPGBC
from .col import CoL
from .dex import DEX

from .sac import SAC
from .sqil import SQIL
from .amp import AMP
from .awac import AWAC
from .double_ddpg import DoubleDDPG
from .dual_ddpg import DualDDPG

# Optional agents that add heavy dependencies; keep them out of the way
# when running standard algorithms like DDPG.
try:
    from .visdex import VisDEX  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    VisDEX = None

try:
    from .e2e import E2EAgent  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    E2EAgent = None


AGENTS = {
    'DDPG': DDPG,
    'DDPGBC': DDPGBC,
    'CoL': CoL,
    'DEX': DEX,
    'SAC': SAC,
    'SQIL': SQIL,
    'AMP': AMP,
    'AWAC': AWAC,
    'DoubleDDPG': DoubleDDPG,
    'DualDDPG': DualDDPG,
}

if VisDEX is not None:
    AGENTS['VisDEX'] = VisDEX
if E2EAgent is not None:
    AGENTS['E2E'] = E2EAgent


def make_agent(env_params, sampler, cfg):
    if cfg.name not in AGENTS.keys():
        assert 'Agent is not supported: %s' % cfg.name
    else:
        return AGENTS[cfg.name](
            env_params=env_params,
            sampler=sampler,
            agent_cfg=cfg
        )
