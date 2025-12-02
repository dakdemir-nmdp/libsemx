import pytest
import pandas as pd
import numpy as np
import semx

def test_gxe_model_builder():
    # Simulate data
    n_gen = 10
    n_env = 3
    genotypes = [f"G{i}" for i in range(n_gen)]
    environments = [f"E{j}" for j in range(n_env)]
    
    data = []
    for e in environments:
        for g in genotypes:
            data.append({
                "yield": np.random.randn(),
                "gid": g,
                "env": e
            })
    df = pd.DataFrame(data)
    
    # Create GRM
    K = np.eye(n_gen)
    
    # Build model
    model, df_aug = semx.gxe_model(
        formula="yield ~ 1",
        data=df,
        genotype="gid",
        environment="env",
        genomic=K
    )
    
    # Check augmented data
    assert "gid:env" in df_aug.columns
    
    # Check model structure
    ir = model.to_ir()
    
    # Check random effects in IR
    # The bindings expose random_effects as a list of RandomEffectSpec objects
    res = ir.random_effects
    assert len(res) == 3
    
    # Check names
    names = [re.id for re in res]
    assert "u_gid" in names
    assert "u_env" in names
    assert "u_gid:env" in names
    
    # Check covariance for u_gid
    re_g = next(r for r in res if r.id == "u_gid")
    assert re_g.covariance_id == "K_g"
    
    # Check covariance for u_env (should be diagonal/default)
    re_e = next(r for r in res if r.id == "u_env")
    assert re_e.covariance_id == "cov_env"
    
    # Check covariance structure in IR
    cov_e = next(c for c in ir.covariances if c.id == "cov_env")
    assert cov_e.structure == "diagonal"

def test_gxe_model_fit_smoke():
    # Smoke test for fitting
    n_gen = 5
    n_env = 2
    genotypes = [f"G{i}" for i in range(n_gen)]
    environments = [f"E{j}" for j in range(n_env)]
    
    data = []
    for e in environments:
        for g in genotypes:
            data.append({
                "y": np.random.randn(),
                "gid": g,
                "env": e
            })
    df = pd.DataFrame(data)
    
    model, df_aug = semx.gxe_model(
        formula="y ~ 1",
        data=df,
        genotype="gid",
        environment="env"
    )
    
    # Fit
    fit = model.fit(df_aug)
    # Smoke test: just ensure it ran and produced a result
    assert fit.optimization_result.iterations >= 0
