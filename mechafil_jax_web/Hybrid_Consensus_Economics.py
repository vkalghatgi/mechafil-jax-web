import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.markdown("[![CryptoEconLab](./app/static/cover.png)](https://cryptoeconlab.io)")

st.sidebar.success("Select a Page above.")

st.markdown(
    """
# Hybrid Consensus Pledge Simulator

This interactive calculator allows you to investigate and forecast network econonoics under a number of parameterizations and implementations of Hybrid Consensus Pledge
(discussed and summarized below)

## Introduction

- Quality adjusted power (QAP) is expected to cross the baseline function on 2024-05-22 (median estimate) with a 90% credible interval between 2023-11-02 and 2025-04-22.
- When this happens, the consensus pledge component of per sector pledge will be divided by the baseline rather than the network QAP, causing Initial Pledge to decrease exponentially. 
- The baseline in Initial Pledge denominator is designed to induce higher onboarding when the network storage is not growing fast enough. However, if onboarding does not respond with annual doubling, then total locked pledge collateral will decrease aggressively, approximately exponentially. This is the baseline pledge bug.

See [Introduction to the Baseline Pledge Bug Document](https://docs.google.com/document/d/1V3Hm1uKemnuVBmdW3KJYqeecA5-A_qf2eQana-9NuUA/edit) for further context. 

- The proposal introduces Hybrid Consensus Pledge model analogous to the Hybrid Minting model in order to mitigate the risks of the current initial pledge model such that consensus pledge is split into a "simple" and "baseline" component:
    - A sector's simple consensus pledge is solely dependent on the sector's QAP with respect to the network's upon sector commit. 
    - A sector's baseline consensus pledge will maintain the current initial pledge property in which it is divided by the maximum of Network QAP and the baseline $b(t)$. 
- Similarly to how simple minting provides a floor on network rewards, simple consensus pledge can provide a reasonable floor on collateral requirements whilst also maintaining onboarding incentives through exponentially decreasing baseline consensus `pledge.

## Current Protocol Specification 

### Initial Pledge

Defining variables:

- $P$: Initial Pledge 
- $P_{s}$: Initial Storage Pledge
- $P_{c}$: Initial Consensus Pledge 
- $Q$: Network QAP 
- $q$: Sector QAP
- $b(t)$: Filecoin's exponential baseline function
- $S$: Filecoin Circulating Supply 
- $\\alpha$ Target lock parameter (currently: $\\alpha = 0.3$)

Filecoin initial pledge for a sector committed at time $t$ with QAP $q$ is as follows: 

$$ P(t) = P_{S}(t) + P_{C}(t) $$

where the initial pledge for a sector with QAP $q$ committed at time $t$ is 

$$ P_{S}(t) = E[20DaysBlockReward] $$

and

$$P_{C}(t) = \\alpha S(t) \\cdot \\frac{q}{max(Q(t), b(t))} $$

### Minting (Brief Overview)

Minting is broken into relevant components as shown below:

- $M_{\infty B}$ is the total number of tokens to be emitted via baseline minting: $M_{\infty B} = M_\infty \\cdot \gamma$. Correspondingly, $M_{\infty S}$ is the total asymptotic number of tokens to be emitted via simple minting: $M_{\infty S} = M_\infty \\cdot (1 - \gamma)$. Of course, $M_{\infty B} + M_{\infty S} = M_\infty$.

- $M_S(t)$ is the total number of tokens that should ideally have been emitted by simple minting up until time $t$. It is defined as $M_S(t) = M_{\infty S} \\cdot (1 - e^{-\lambda t})$. It is easy to verify that $\\lim_{t\\rightarrow\\infty} M_S(t) = M_{\infty S}$.

where $\\gamma = 0.7$ per the current protocol spec
## Proposed Hybrid Consensus Pledge Model 

Propose adjusting the initial consensus pledge calculation such that it has a "simple" consensus pledge $P_{C,S}$ and "baseline" consensus pledge component $P_{C,B}$, determined by some weighting parameterized by $\\gamma'$, analogous to the minting model. This can be implemented with a number of weighting schemes that interpolate differently between the regions $0 \le \\gamma' \le 1$. Some of these weighting schemas are specified below: 

### Arithmetic Weighting 

This is most analogous to the current minting model and $P_{C}$ is calculated as per below: 

$$P_C(t) = (1-\\gamma')P_{C,S}(t) + \\gamma' P_{C,B}(t) $$

Initially, set $\\gamma' = 0.7$: (Note, $\\gamma$ can take on any value such that $0 \le \\gamma' \le 1$, but start with $\\gamma' = \\gamma$ per the minting model's specification:  

$$ P_{C,S}(t) = \\alpha S(t) \\cdot \\frac{q}{Q(t)} $$
$$ P_{C,B}(t) = \\alpha S(t) \\cdot \\frac{q}{max(Q(t), b(t))} $$

$P_C(t)$ reduces to: 

$$ P_C(t) = \\alpha S(t) \\cdot q \\left( (1 - \\gamma') \\frac{1}{Q(t)} + \\gamma'
\\frac{1}{max(Q(t), b(t))} \\right) $$

$-\\frac{0.3 \\gamma  q(t) S(t) b'(t)}{b(t)^2}+\\frac{0.3 \\gamma  S(t) q'(t)}{b(t)}+\\frac{0.3 \\gamma  q(t) S'(t)}{b(t)}+\\frac{0.3 (1-\\gamma ) S(t) q'(t)}{Q(t)}-\\frac{0.3 (1-\\gamma ) q(t) S(t) Q'(t)}{Q(t)^2}+\\frac{0.3 (1-\\gamma ) q(t) S'(t)}{Q(t)}$

### Geometric Weighting 

$$ P_{C}(t) = P_{C,S}(t)^{1 - \\gamma'} \\cdot P_{C,B}(t)^{\\gamma'} $$

### Harmonic Weighting 

$$ P_{C}(t) = \\frac{1}{(1 - \\gamma')\\frac{1}{P_{C,S}(t)} + \\gamma' \\frac{1}{P_{C,B}(t)}} $$

# Want to learn more?

- Check out [CryptoEconLab](https://cryptoeconlab.io)

- Engage with us on [X](https://x.com/cryptoeconlab)

- Read more of our research on [Medium](https://medium.com/cryptoeconlab) and [HackMD](https://hackmd.io/@cryptoecon/almanac/)

# Disclaimer
CryptoEconLab designed this application for informational purposes only. CryptoEconLab does not provide legal, tax, financial or investment advice. No party should act in reliance upon, or with the expectation of, any such advice.
"""
)