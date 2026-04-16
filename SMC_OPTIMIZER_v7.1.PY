"""
SMC OPTIMIZER v7 - NIVEL INSTITUCIONAL
WDO Dolar Mini B3

BIBLIOTECAS UTILIZADAS:

- skfolio: CombinatorialPurgedCV (CPCV)
- arch: CircularBlockBootstrap (Block Bootstrap)
- optuna: Bayesian optimization (TPE)
- scipy: DSR, PBO, estatisticas
- smartmoneyconcepts: indicadores SMC oficiais

CORRECOES v7:

1. Swing lookahead: sinais usados em i + swing_length
2. Filtro HTF 15min obrigatorio
3. Grid 72 combos logicos para WDO
4. Min 300 trades
5. CPCV com purging e embargo
6. DSR corrige Sharpe para multiplos testes
7. PBO via CSCV
8. Block Bootstrap preserva estrutura temporal
9. Optuna Bayesian search em vez de grid bruto
10. Score por robustez OOS
11. Stress test
12. Signal Quality Analysis
"""

import pandas as pd
import numpy as np
import json
import sys
import os
import time
import warnings
import itertools
import math
from datetime import datetime
from multiprocessing import Pool, cpu_count
from scipy import stats
import pickle
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

CSV_PATH = "/workspace/strategy_composer/wdo_clean.csv"
OUTPUT_DIR = "/workspace/param_opt_output"
N_CORES = min(16, cpu_count())
CAPITAL = 50_000.0
MULT_WDO = 10.0
COMISSAO = 5.0
SLIPPAGE = 2.0
CLOSE_BREAK = True

# Espaco de busca para Optuna (valores logicos para WDO)
SEARCH_SPACE = {
    "rr_min": [2.0, 2.5, 3.0],
    "atr_mult_sl": [0.7, 1.0, 1.5],
    "swing_length": [5, 7],
    "estrategia": [1, 2, 3, 5],
}

# 3x3x2x4 = 72 combos
MIN_TRADES = 300
MAX_PF = 3.0
MAX_SHARPE = 4.0
MAX_DD = -25.0
N_OPTUNA_TRIALS = 72  # testa todos os combos via Optuna

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# SECAO 1: DSR - Deflated Sharpe Ratio
# Bailey & Lopez de Prado (2014)
# ================================================================

def calcular_dsr(sharpe_obs, n_trials, T, skew=0.0, kurt=3.0):
    """
    Corrige Sharpe para selecao bias e nao-normalidade.
    n_trials: quantas configs foram testadas
    T: numero de trades/retornos
    Retorna: (dsr_valor, interpretacao)
    """
    if n_trials <= 1 or T < 30 or sharpe_obs <= 0:
        return 0.0, "INSUFICIENTE"

    # Expected maximum Sharpe sob H0 (nenhuma estrategia tem edge)
    # Formula exata de Bailey & Lopez de Prado eq. 8
    gamma = 0.5772156649  # constante de Euler-Mascheroni
    e_max_sr = (
        (1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
        + gamma * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
    )

    # Variancia do Sharpe ajustada para nao-normalidade (eq. 4)
    var_sr = (
        (1.0 / T)
        * (
            1
            + 0.5 * sharpe_obs**2
            - skew * sharpe_obs
            + (kurt - 3) / 4.0 * sharpe_obs**2
        )
    )

    if var_sr <= 0:
        return 0.0, "ERRO_VARIANCIA"

    # PSR: Probabilistic Sharpe Ratio
    # P(SR_verdadeiro > 0 | SR_observado)
    _psr = float(
        stats.norm.cdf(
            (sharpe_obs * np.sqrt(T))
            / np.sqrt(
                1.0
                + 0.5 * sharpe_obs**2
                - skew * sharpe_obs
                + (kurt - 3) / 4.0 * sharpe_obs**2
            )
        )
    )

    # DSR: desconta pelo numero de trials (eq. 10)
    dsr_stat = (sharpe_obs - e_max_sr) / np.sqrt(var_sr)
    dsr = float(stats.norm.cdf(dsr_stat))

    if dsr >= 0.95:
        interp = "EXCELENTE (p>0.95)"
    elif dsr >= 0.85:
        interp = "BOM (p>0.85)"
    elif dsr >= 0.70:
        interp = "ACEITAVEL (p>0.70)"
    else:
        interp = "FRACO - provavel overfitting"

    return round(dsr, 4), interp


# ================================================================
# SECAO 2: PBO - Probability of Backtest Overfitting
# Bailey, Borwein, Lopez de Prado, Zhu (2015)
# Metodo CSCV simplificado
# ================================================================

def calcular_pbo(retornos_por_config, n_splits=8):
    """
    retornos_por_config: dict {config_id: array_de_retornos}
    Retorna PBO entre 0 (sem overfitting) e 1 (overfitting total)
    """
    configs = list(retornos_por_config.keys())
    N = len(configs)
    if N < 4:
        return 0.5, "POUCOS CONFIGS"

    # Alinhar retornos no mesmo comprimento
    min_len = min(len(v) for v in retornos_por_config.values())
    if min_len < n_splits * 4:
        return 0.5, "POUCOS DADOS"

    # Montar matriz T x N
    T = (min_len // n_splits) * n_splits
    matriz = np.zeros((T, N))
    for j, k in enumerate(configs):
        matriz[:, j] = retornos_por_config[k][:T]

    split_size = T // n_splits
    splits = [matriz[s * split_size:(s + 1) * split_size, :] for s in range(n_splits)]

    # CSCV: gerar combinacoes IS/OOS simetricas
    logits = []
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    np.random.seed(42)
    if len(combos) > 100:
        idx = np.random.choice(len(combos), 100, replace=False)
        combos = [combos[i] for i in idx]

    sharpe_fn = lambda x: x.mean(axis=0) / (x.std(axis=0) + 1e-9)

    for is_idx in combos:
        oos_idx = [i for i in range(n_splits) if i not in is_idx]

        is_data = np.vstack([splits[i] for i in is_idx])
        oos_data = np.vstack([splits[i] for i in oos_idx])

        is_perf = sharpe_fn(is_data)
        oos_perf = sharpe_fn(oos_data)

        best_is = np.argmax(is_perf)
        oos_rank = stats.rankdata(oos_perf)[best_is]
        n_configs = N

        # Logit de P(melhor IS < mediana OOS)
        rank_norm = oos_rank / (n_configs + 1)
        logit = np.log(rank_norm / (1 - rank_norm + 1e-9))
        logits.append(logit)

    if not logits:
        return 0.5, "SEM DADOS"

    pbo = float(np.mean(np.array(logits) <= 0))

    if pbo <= 0.10:
        interp = "EXCELENTE - baixo overfitting"
    elif pbo <= 0.25:
        interp = "BOM - risco moderado"
    elif pbo <= 0.40:
        interp = "ATENCAO - risco elevado"
    else:
        interp = "ALTO RISCO DE OVERFITTING"

    return round(pbo, 4), interp


# ================================================================
# SECAO 3: BLOCK BOOTSTRAP (arch.CircularBlockBootstrap)
# ================================================================

def monte_carlo_block_bootstrap(trades, n_sim=2000, capital=CAPITAL):
    """
    Monte Carlo com Circular Block Bootstrap (arch library).
    Preserva dependencia temporal dos trades.
    """
    from arch.bootstrap import CircularBlockBootstrap

    fechados = [t for t in trades if t.get("resultado") in ("WIN", "LOSS")]
    if len(fechados) < 20:
        return {}

    pnls = np.array([t["pnl_brl"] for t in fechados])
    n = len(pnls)
    block_size = max(5, int(np.sqrt(n)))

    print(f"\n[MC] Block Bootstrap (arch): {n_sim:,} sims | bloco={block_size}")

    bs = CircularBlockBootstrap(block_size, pnls, seed=42)

    rets, dds, ruinas = [], [], 0
    for data, _ in bs.bootstrap(n_sim):
        sim_pnl = data[0]
        eq = np.insert(capital + np.cumsum(sim_pnl), 0, capital)
        pk = np.maximum.accumulate(eq)
        dd = float(((eq - pk) / pk * 100).min())
        r = float((eq[-1] - capital) / capital * 100)
        rets.append(r)
        dds.append(dd)
        if eq[-1] < capital * 0.5:
            ruinas += 1

    rf, md = np.array(rets), np.array(dds)
    skew_pnl = float(stats.skew(pnls))
    kurt_pnl = float(stats.kurtosis(pnls) + 3)

    res = {
        "metodo": "Circular Block Bootstrap (arch)",
        "block_size": block_size,
        "n_simulacoes": n_sim,
        "prob_lucro_pct": round(float((rf > 0).mean() * 100), 1),
        "retorno_mediana": round(float(np.median(rf)), 2),
        "retorno_p10": round(float(np.percentile(rf, 10)), 2),
        "retorno_p25": round(float(np.percentile(rf, 25)), 2),
        "retorno_p75": round(float(np.percentile(rf, 75)), 2),
        "retorno_p90": round(float(np.percentile(rf, 90)), 2),
        "dd_mediano": round(float(np.median(md)), 2),
        "dd_p90": round(float(np.percentile(md, 90)), 2),
        "dd_pior": round(float(md.min()), 2),
        "prob_ruina_pct": round(float(ruinas / n_sim * 100), 2),
        "prob_dd_10": round(float((md < -10).mean() * 100), 1),
        "prob_dd_20": round(float((md < -20).mean() * 100), 1),
        "skewness_pnl": round(skew_pnl, 3),
        "kurtosis_pnl": round(kurt_pnl, 3),
    }

    print(
        f"[MC] OK Prob.lucro:{res['prob_lucro_pct']}% | "
        f"DD:{res['dd_mediano']}% | Ruina:{res['prob_ruina_pct']}%"
    )
    return res


# ================================================================
# SECAO 4: CPCV (skfolio.CombinatorialPurgedCV)
# ================================================================

def validar_cpcv(df, config, n_folds=6, n_test_folds=2, purge_size=30, embargo_size=10):
    """
    Combinatorial Purged Cross-Validation com skfolio.
    purge_size: candles a purgar entre treino e teste
    embargo_size: candles de embargo apos o teste
    Retorna: metricas por path e taxa de sucesso
    """
    from skfolio.model_selection import CombinatorialPurgedCV

    print(
        f"\n[CPCV] {n_folds} folds / {n_test_folds} test | "
        f"purge={purge_size} embargo={embargo_size}"
    )

    X = np.arange(len(df)).reshape(-1, 1)
    cv = CombinatorialPurgedCV(
        n_folds=n_folds,
        n_test_folds=n_test_folds,
        purged_size=purge_size,
        embargo_size=embargo_size,
    )

    resultados = []
    retornos_oos = []

    for split_idx, (train_idx, test_paths) in enumerate(cv.split(X)):
        df_tr = df.iloc[train_idx]
        for path_idx, test_idx in enumerate(test_paths):
            df_te = df.iloc[test_idx]
            if len(df_tr) < 500 or len(df_te) < 100:
                continue
            try:
                sw = config["swing_length"]
                bt_params = {
                    k: v
                    for k, v in config.items()
                    if k in ["rr_min", "atr_mult_sl", "poi_janela", "choch_janela", "estrategia"]
                }

                df_tr_ind = preparar_smc_htf(df_tr, swing_length=sw)
                df_te_ind = preparar_smc_htf(df_te, swing_length=sw)

                _, tr_eq = backtest(df_tr_ind, **bt_params)
                te_t, te_eq = backtest(df_te_ind, **bt_params)
                m_te = metricas(te_t, te_eq)

                if m_te:
                    pnl = m_te["total_pnl_brl"]
                    retornos_oos.extend(
                        [t["pnl_brl"] for t in te_t if t.get("resultado") in ("WIN", "LOSS")]
                    )
                    resultados.append(
                        {
                            "split": split_idx,
                            "path": path_idx,
                            "pnl": pnl,
                            "pf": m_te["profit_factor"],
                            "trades": m_te["total_trades"],
                        }
                    )
                    print(
                        f"  Split {split_idx} Path {path_idx}: "
                        f"PF={m_te['profit_factor']:.3f} "
                        f"Trades={m_te['total_trades']} "
                        f"PnL=R${pnl:,.0f}"
                    )
            except Exception:
                pass

    if not resultados:
        return {}, [], 0.0

    lucrativos = sum(1 for r in resultados if r["pnl"] > 0)
    taxa = lucrativos / len(resultados) if resultados else 0
    print(f"\n[CPCV] {lucrativos}/{len(resultados)} paths lucrativos ({taxa*100:.0f}%)")

    return resultados, retornos_oos, taxa


# ================================================================
# SECAO 5: CARREGAMENTO E PREPARACAO DOS DADOS
# ================================================================

def carregar():
    print(f"[DATA] Carregando {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df.dropna()
    df = df[df["close"] > 0]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"[DATA] OK {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def resample_htf(df_1min, tf_minutes=15):
    """Agrega 1min para HTF (15min)."""
    rule = f"{tf_minutes}min"
    htf = df_1min.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    return htf


def preparar_smc_htf(df_1min, swing_length=5):
    """
    CORRECAO CRITICA v7:
    1. Calcula indicadores SMC no 1min
    2. Calcula bias HTF no 15min
    3. Sinais so validos apos i + swing_length (sem lookahead)
    4. Adiciona filtro HTF obrigatorio
    """
    from smartmoneyconcepts import smc as SMC

    # === HTF 15min para bias ===
    df_15 = resample_htf(df_1min, 15)
    sw_15 = SMC.swing_highs_lows(df_15, swing_length=swing_length)
    bs_15 = SMC.bos_choch(df_15, sw_15, close_break=CLOSE_BREAK)

    choch_15 = bs_15["CHOCH"].fillna(0)
    htf_bias = pd.Series(0, index=df_15.index)
    ultimo = 0
    for i, val in choch_15.items():
        if val != 0:
            ultimo = int(val)
        htf_bias[i] = ultimo

    # Garantir que ambos os indices estao ordenados antes do reindex
    htf_bias = htf_bias.sort_index()
    df_1min_sorted_idx = df_1min.index.sort_values()
    htf_bias_1min = htf_bias.reindex(df_1min_sorted_idx, method="ffill").fillna(0)
    htf_bias_1min = htf_bias_1min.reindex(df_1min.index).fillna(0)

    # === 1min indicadores ===
    swings = SMC.swing_highs_lows(df_1min, swing_length=swing_length)
    estrutura = SMC.bos_choch(df_1min, swings, close_break=CLOSE_BREAK)
    fvg = SMC.fvg(df_1min)
    ob = SMC.ob(df_1min, swings)

    try:
        liq = SMC.liquidity(df_1min, swings)
        liq_v = liq["Liquidity"].values if "Liquidity" in liq.columns else np.zeros(len(df_1min))
        liq_sw = liq["Swept"].values if "Swept" in liq.columns else np.full(len(df_1min), np.nan)
    except Exception:
        liq_v = np.zeros(len(df_1min))
        liq_sw = np.full(len(df_1min), np.nan)

    h, l, cp = df_1min["high"], df_1min["low"], df_1min["close"].shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    sh = df_1min["high"].rolling(50).max()
    sl2 = df_1min["low"].rolling(50).min()
    mid = sl2 + (sh - sl2) * 0.5

    r = df_1min.copy()

    choch_raw = pd.to_numeric(
        pd.Series(estrutura["CHOCH"].values, index=df_1min.index),
        errors="coerce",
    ).fillna(0)
    r["choch"] = choch_raw.shift(swing_length).fillna(0)

    fvg_raw = pd.to_numeric(
        pd.Series(fvg["FVG"].values, index=df_1min.index),
        errors="coerce",
    ).fillna(0)
    r["fvg"] = fvg_raw.shift(swing_length).fillna(0)
    r["fvg_top"] = pd.Series(fvg["Top"].values, index=df_1min.index).shift(swing_length)
    r["fvg_bot"] = pd.Series(fvg["Bottom"].values, index=df_1min.index).shift(swing_length)

    ob_raw = pd.to_numeric(
        pd.Series(ob["OB"].values, index=df_1min.index),
        errors="coerce",
    ).fillna(0)
    r["ob"] = ob_raw.shift(swing_length).fillna(0)
    r["ob_top"] = pd.Series(ob["Top"].values, index=df_1min.index).shift(swing_length)
    r["ob_bot"] = pd.Series(ob["Bottom"].values, index=df_1min.index).shift(swing_length)

    r["liq"] = pd.Series(liq_v, index=df_1min.index).shift(swing_length).fillna(0)
    r["liq_sw"] = pd.Series(liq_sw, index=df_1min.index).shift(swing_length)
    r["atr"] = atr
    r["premium"] = (df_1min["close"] > mid).fillna(False).astype(float)
    r["discount"] = (df_1min["close"] < mid).fillna(False).astype(float)
    r["london"] = ((df_1min.index.hour >= 9) & (df_1min.index.hour < 12)).astype(float)
    r["ny"] = ((df_1min.index.hour >= 13) & (df_1min.index.hour < 17)).astype(float)

    r["htf_bias"] = htf_bias_1min.values
    return r


# ================================================================
# SECAO 6: ENGINE DE BACKTEST
# ================================================================

def backtest(df_ind, rr_min=2.0, atr_mult_sl=1.0, poi_janela=20, choch_janela=30, estrategia=1, capital=CAPITAL):
    """
    Backtest com entrada no OPEN do proximo candle.
    Filtro HTF obrigatorio: so opera na direcao do bias 15min.
    """
    trades = []
    equity = [capital]
    cap = capital
    em_pos = False
    trade = None

    fvgs_bull, fvgs_bear = [], []
    obs_bull, obs_bear = [], []
    ult_choch_bull = ult_choch_bear = -9999
    ult_liq_bull = ult_liq_bear = -9999
    sinal_pendente = poi_pendente = None

    arr = df_ind.values
    cols = {c: i for i, c in enumerate(df_ind.columns)}

    def v(row, col):
        return row[cols[col]]

    for i in range(50, len(df_ind)):
        row = arr[i]

        # === EXECUCAO: OPEN do proximo candle ===
        if sinal_pendente is not None and not em_pos:
            sinal = sinal_pendente
            poi = poi_pendente
            sinal_pendente = poi_pendente = None

            entry = float(v(row, "open"))
            atr_e = float(v(arr[i - 1], "atr"))
            if np.isnan(atr_e) or atr_e <= 0:
                atr_e = 5.0

            sl = poi["bot"] - atr_e * atr_mult_sl if sinal == 1 else poi["top"] + atr_e * atr_mult_sl

            risk = abs(entry - sl)
            if risk > 0:
                tp = entry + sinal * risk * rr_min
                rr_r = abs(tp - entry) / risk
                if rr_r >= rr_min * 0.95 and risk * MULT_WDO / cap <= 0.05:
                    em_pos = True
                    trade = {
                        "entry_dt": str(df_ind.index[i])[:16],
                        "d": sinal,
                        "entry": round(entry, 2),
                        "sl": round(sl, 2),
                        "tp": round(tp, 2),
                        "rr": round(rr_r, 2),
                        "poi_tipo": poi.get("tipo", "?"),
                        "estrategia": estrategia,
                        "capital_pre": round(cap, 2),
                    }

        # === GERENCIAR POSICAO ===
        if em_pos and trade:
            d, sl, tp, en = trade["d"], trade["sl"], trade["tp"], trade["entry"]
            lo = float(v(row, "low"))
            hi = float(v(row, "high"))
            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)
            if hit_sl or hit_tp:
                saida = sl if hit_sl else tp
                pts = (saida - en) * d
                brl = pts * MULT_WDO - COMISSAO - SLIPPAGE * MULT_WDO * 0.5
                cap += brl
                equity.append(round(cap, 2))
                trade.update(
                    {
                        "saida": round(saida, 2),
                        "pnl_pts": round(pts, 2),
                        "pnl_brl": round(brl, 2),
                        "resultado": "WIN" if hit_tp else "LOSS",
                        "saida_dt": str(df_ind.index[i])[:16],
                    }
                )
                trades.append(trade)
                em_pos = False
                trade = None
            continue

        # === COLETAR SINAIS ===
        htf_bias = int(v(row, "htf_bias"))
        choch = float(v(row, "choch"))

        if choch == 1:
            ult_choch_bull = i
            fvgs_bull.clear()
            obs_bull.clear()
        elif choch == -1:
            ult_choch_bear = i
            fvgs_bear.clear()
            obs_bear.clear()

        liq_val = float(v(row, "liq"))
        liq_sw = v(row, "liq_sw")
        if not np.isnan(liq_sw):
            if liq_val == -1:
                ult_liq_bull = i
            if liq_val == 1:
                ult_liq_bear = i

        fvg_v = float(v(row, "fvg"))
        if fvg_v == 1 and not np.isnan(v(row, "fvg_top")):
            fvgs_bull.append(
                {"top": float(v(row, "fvg_top")), "bot": float(v(row, "fvg_bot")), "tipo": "FVG"}
            )
        elif fvg_v == -1 and not np.isnan(v(row, "fvg_top")):
            fvgs_bear.append(
                {"top": float(v(row, "fvg_top")), "bot": float(v(row, "fvg_bot")), "tipo": "FVG"}
            )

        ob_v = float(v(row, "ob"))
        if ob_v == 1 and not np.isnan(v(row, "ob_top")):
            obs_bull.append(
                {"top": float(v(row, "ob_top")), "bot": float(v(row, "ob_bot")), "tipo": "OB"}
            )
        elif ob_v == -1 and not np.isnan(v(row, "ob_top")):
            obs_bear.append(
                {"top": float(v(row, "ob_top")), "bot": float(v(row, "ob_bot")), "tipo": "OB"}
            )

        fvgs_bull = fvgs_bull[-poi_janela:]
        fvgs_bear = fvgs_bear[-poi_janela:]
        obs_bull = obs_bull[-poi_janela:]
        obs_bear = obs_bear[-poi_janela:]

        close = float(v(row, "close"))
        premium = float(v(row, "premium")) == 1
        discount = float(v(row, "discount")) == 1
        london = float(v(row, "london")) == 1
        ny = float(v(row, "ny")) == 1

        # Filtros de estrategia
        if estrategia == 2:
            pode_bull = discount and (htf_bias >= 0)
            pode_bear = premium and (htf_bias <= 0)
        elif estrategia == 3:
            pode_bull = (london or ny) and (htf_bias >= 0)
            pode_bear = (london or ny) and (htf_bias <= 0)
        elif estrategia == 5:
            pode_bull = ((i - ult_liq_bull) <= choch_janela) and (htf_bias >= 0)
            pode_bear = ((i - ult_liq_bear) <= choch_janela) and (htf_bias <= 0)
        else:
            pode_bull = htf_bias >= 0
            pode_bear = htf_bias <= 0

        sinal = poi = None

        if (i - ult_choch_bull) <= choch_janela and pode_bull:
            for p in reversed(fvgs_bull + obs_bull):
                if close <= p["top"] and close >= p["bot"] * 0.998:
                    sinal = 1
                    poi = p
                    break

        if sinal is None and (i - ult_choch_bear) <= choch_janela and pode_bear:
            for p in reversed(fvgs_bear + obs_bear):
                if close >= p["bot"] and close <= p["top"] * 1.002:
                    sinal = -1
                    poi = p
                    break

        if sinal is not None and poi is not None and i < len(df_ind) - 1:
            sinal_pendente = sinal
            poi_pendente = poi

    if em_pos and trade:
        last = float(arr[-1][cols["close"]])
        pts = (last - trade["entry"]) * trade["d"]
        brl = pts * MULT_WDO - COMISSAO
        cap += brl
        trade.update(
            {
                "saida": last,
                "pnl_pts": round(pts, 2),
                "pnl_brl": round(brl, 2),
                "resultado": "ABERTO",
                "saida_dt": str(df_ind.index[-1])[:16],
            }
        )
        trades.append(trade)
        equity.append(round(cap, 2))

    return trades, equity


# ================================================================
# SECAO 7: METRICAS
# ================================================================

def metricas(trades, equity, capital=CAPITAL, n_trials=1):
    fechados = [t for t in trades if t.get("resultado") in ("WIN", "LOSS")]
    n = len(fechados)
    if n < MIN_TRADES:
        return None

    df_t = pd.DataFrame(fechados)
    wins = df_t[df_t["resultado"] == "WIN"]
    loses = df_t[df_t["resultado"] == "LOSS"]
    wr = len(wins) / n * 100
    avg_w = float(wins["pnl_brl"].mean()) if len(wins) else 0
    avg_l = float(loses["pnl_brl"].mean()) if len(loses) else -1
    pf = abs(float(wins["pnl_brl"].sum()) / float(loses["pnl_brl"].sum())) if float(loses["pnl_brl"].sum()) != 0 else 9999
    pnl = float(df_t["pnl_brl"].sum())

    eq = pd.Series(equity)
    peak = eq.cummax()
    mdd = float(((eq - peak) / peak * 100).min())

    rets = eq.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252 * 390)) if rets.std() > 0 else 0
    neg = rets[rets < 0]
    sortino = float(rets.mean() / neg.std() * np.sqrt(252 * 390)) if len(neg) > 1 else 0

    # Filtros anti-overfitting
    if pf > MAX_PF:
        return None
    if sharpe > MAX_SHARPE:
        return None
    if mdd < MAX_DD:
        return None

    pnl_arr = df_t["pnl_brl"].values
    skew_pnl = float(stats.skew(pnl_arr))
    kurt_pnl = float(stats.kurtosis(pnl_arr) + 3)

    dsr_val, dsr_interp = calcular_dsr(
        sharpe_obs=sharpe,
        n_trials=n_trials,
        T=n,
        skew=skew_pnl,
        kurt=kurt_pnl,
    )

    pnls = df_t["pnl_brl"].values
    janelas_pos = sum(1 for s in range(0, n - 50, 25) if pnls[s:s + 50].sum() > 0)
    n_janelas = max(1, len(range(0, n - 50, 25)))
    pct_jan = janelas_pos / n_janelas * 100

    calmar = (pnl / capital) / abs(mdd / 100) if mdd != 0 else 0

    return {
        "total_trades": n,
        "wins": int(len(wins)),
        "losses": int(len(loses)),
        "win_rate": round(wr, 2),
        "profit_factor": round(pf, 3),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "dsr": dsr_val,
        "dsr_interp": dsr_interp,
        "avg_win_brl": round(avg_w, 2),
        "avg_loss_brl": round(avg_l, 2),
        "expectancy_brl": round((wr / 100 * avg_w) + ((1 - wr / 100) * avg_l), 2),
        "total_pnl_brl": round(pnl, 2),
        "retorno_pct": round(pnl / capital * 100, 2),
        "max_drawdown_pct": round(mdd, 2),
        "capital_final": round(capital + pnl, 2),
        "pct_janelas_positivas": round(pct_jan, 1),
        "skewness_pnl": round(skew_pnl, 3),
        "kurtosis_pnl": round(kurt_pnl, 3),
        "poi_tipos": df_t["poi_tipo"].value_counts().to_dict(),
        "por_estrategia": df_t["estrategia"].value_counts().to_dict(),
    }


def stress_test(trades, equity, capital=CAPITAL):
    """
    Stress tests:
    1. Sem top 5 trades (estrategia fragil se depender deles)
    2. Slippage 2x
    3. Comissao +50%
    """
    fechados = [t for t in trades if t.get("resultado") in ("WIN", "LOSS")]
    if len(fechados) < 20:
        return {}

    resultados = {}

    df_t = pd.DataFrame(fechados)
    top5 = df_t.nlargest(5, "pnl_brl").index
    sem_top5 = df_t.drop(top5)
    pnl_sem_top5 = sem_top5["pnl_brl"].sum()
    resultados["sem_top5_pnl"] = round(float(pnl_sem_top5), 2)
    resultados["sem_top5_lucrativo"] = bool(pnl_sem_top5 > 0)

    pnl_slip2x = sum(t["pnl_brl"] - SLIPPAGE * MULT_WDO * 0.5 for t in fechados)
    resultados["slippage_2x_pnl"] = round(float(pnl_slip2x), 2)
    resultados["slippage_2x_lucrativo"] = bool(pnl_slip2x > 0)

    pnl_com50 = sum(t["pnl_brl"] - COMISSAO * 0.5 for t in fechados)
    resultados["comissao_50pct_pnl"] = round(float(pnl_com50), 2)
    resultados["comissao_50pct_lucrativo"] = bool(pnl_com50 > 0)

    n_pass = sum(
        [
            resultados["sem_top5_lucrativo"],
            resultados["slippage_2x_lucrativo"],
            resultados["comissao_50pct_lucrativo"],
        ]
    )
    resultados["stress_score"] = f"{n_pass}/3"
    resultados["aprovado"] = n_pass >= 2

    return resultados


def signal_quality(df_ind, estrategia=1, choch_janela=30, poi_janela=20):
    """
    Analisa qualidade dos sinais independente da execucao.
    Pergunta: dos sinais gerados, quantos % teriam atingido TP?
    """
    arr = df_ind.values
    cols = {c: i for i, c in enumerate(df_ind.columns)}

    def v(row, col):
        return row[cols[col]]

    sinais_total = 0
    sinais_tp = 0
    sinais_sl = 0

    fvgs_bull, fvgs_bear = [], []
    obs_bull, obs_bear = [], []
    ult_choch_bull = ult_choch_bear = -9999

    for i in range(50, len(df_ind) - 10):
        row = arr[i]

        choch = float(v(row, "choch"))
        if choch == 1:
            ult_choch_bull = i
            fvgs_bull.clear()
            obs_bull.clear()
        elif choch == -1:
            ult_choch_bear = i
            fvgs_bear.clear()
            obs_bear.clear()

        fvg_v = float(v(row, "fvg"))
        if fvg_v == 1 and not np.isnan(v(row, "fvg_top")):
            fvgs_bull.append({"top": float(v(row, "fvg_top")), "bot": float(v(row, "fvg_bot"))})
        elif fvg_v == -1 and not np.isnan(v(row, "fvg_top")):
            fvgs_bear.append({"top": float(v(row, "fvg_top")), "bot": float(v(row, "fvg_bot"))})

        ob_v = float(v(row, "ob"))
        if ob_v == 1 and not np.isnan(v(row, "ob_top")):
            obs_bull.append({"top": float(v(row, "ob_top")), "bot": float(v(row, "ob_bot"))})
        elif ob_v == -1 and not np.isnan(v(row, "ob_top")):
            obs_bear.append({"top": float(v(row, "ob_top")), "bot": float(v(row, "ob_bot"))})

        fvgs_bull = fvgs_bull[-poi_janela:]
        fvgs_bear = fvgs_bear[-poi_janela:]
        obs_bull = obs_bull[-poi_janela:]
        obs_bear = obs_bear[-poi_janela:]

        close = float(v(row, "close"))
        htf_bias = int(v(row, "htf_bias"))
        sinal = poi = None

        if (i - ult_choch_bull) <= choch_janela and htf_bias >= 0:
            for p in reversed(fvgs_bull + obs_bull):
                if close <= p["top"] and close >= p["bot"] * 0.998:
                    sinal = 1
                    poi = p
                    break

        if sinal is None and (i - ult_choch_bear) <= choch_janela and htf_bias <= 0:
            for p in reversed(fvgs_bear + obs_bear):
                if close >= p["bot"] and close <= p["top"] * 1.002:
                    sinal = -1
                    poi = p
                    break

        if sinal is None or poi is None:
            continue

        sinais_total += 1
        atr_v = float(v(row, "atr"))
        if np.isnan(atr_v) or atr_v <= 0:
            atr_v = 5.0

        entry = float(v(arr[min(i + 1, len(arr) - 1)], "open"))
        sl = poi["bot"] - atr_v if sinal == 1 else poi["top"] + atr_v
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        tp = entry + sinal * risk * 2.0

        for j in range(i + 1, min(i + 51, len(arr))):
            hi = float(v(arr[j], "high"))
            lo = float(v(arr[j], "low"))
            if sinal == 1:
                if lo <= sl:
                    sinais_sl += 1
                    break
                if hi >= tp:
                    sinais_tp += 1
                    break
            else:
                if hi >= sl:
                    sinais_sl += 1
                    break
                if lo <= tp:
                    sinais_tp += 1
                    break

    total_resolvidos = sinais_tp + sinais_sl
    sq_wr = sinais_tp / total_resolvidos * 100 if total_resolvidos > 0 else 0

    return {
        "sinais_gerados": sinais_total,
        "sinais_tp": sinais_tp,
        "sinais_sl": sinais_sl,
        "sinais_resolvidos": total_resolvidos,
        "signal_quality_wr": round(sq_wr, 2),
    }


# ================================================================
# SECAO 8: OTIMIZACAO COM OPTUNA (Bayesian TPE)
# ================================================================

_DF_IND_CACHE = {}
_DF_INS_GLOBAL = None


def preparar_cache(df_ins):
    """Pre-computa indicadores para cada combo de swing_length."""
    global _DF_IND_CACHE, _DF_INS_GLOBAL
    _DF_INS_GLOBAL = df_ins
    combos = list(itertools.product(SEARCH_SPACE["swing_length"], [True]))
    print(f"\n[PRE] Pre-computando {len(combos)} conjuntos de indicadores...")
    t0 = time.time()
    for sw, cb in combos:
        key = (sw, cb)
        print(f"  swing={sw}...", end=" ", flush=True)
        t1 = time.time()
        _DF_IND_CACHE[key] = preparar_smc_htf(df_ins, swing_length=sw)
        print(f"OK ({time.time()-t1:.1f}s)")
    print(f"[PRE] OK {time.time()-t0:.1f}s\n")


def objective_optuna(trial):
    """Funcao objetivo para o Optuna."""
    rr = trial.suggest_categorical("rr_min", SEARCH_SPACE["rr_min"])
    am = trial.suggest_categorical("atr_mult_sl", SEARCH_SPACE["atr_mult_sl"])
    sw = trial.suggest_categorical("swing_length", SEARCH_SPACE["swing_length"])
    est = trial.suggest_categorical("estrategia", SEARCH_SPACE["estrategia"])

    key = (sw, True)
    if key not in _DF_IND_CACHE:
        return 0.0

    df_ind = _DF_IND_CACHE[key]
    try:
        trades, equity = backtest(
            df_ind,
            rr_min=rr,
            atr_mult_sl=am,
            poi_janela=20,
            choch_janela=30,
            estrategia=est,
        )
        m = metricas(trades, equity, n_trials=N_OPTUNA_TRIALS)
        if not m:
            return 0.0

        pf_s = min(m["profit_factor"], MAX_PF) / MAX_PF
        dsr_s = m["dsr"]
        exp_s = max(0, min(m["expectancy_brl"], 200)) / 200
        jan_s = m["pct_janelas_positivas"] / 100
        tr_s = min(m["total_trades"], 1000) / 1000

        score = pf_s * 0.25 + dsr_s * 0.25 + exp_s * 0.25 + jan_s * 0.15 + tr_s * 0.10

        trial.set_user_attr("total_trades", m["total_trades"])
        trial.set_user_attr("profit_factor", m["profit_factor"])
        trial.set_user_attr("sharpe_ratio", m["sharpe_ratio"])
        trial.set_user_attr("win_rate", m["win_rate"])
        trial.set_user_attr("dsr", m["dsr"])
        trial.set_user_attr("expectancy_brl", m["expectancy_brl"])
        trial.set_user_attr("max_drawdown", m["max_drawdown_pct"])
        trial.set_user_attr("janelas_pos", m["pct_janelas_positivas"])

        return score
    except Exception:
        return 0.0


def otimizar_optuna(n_trials=72):
    """Roda otimizacao Bayesiana com Optuna TPE."""
    print(f"\n[OPTUNA] Bayesian TPE Optimization: {n_trials} trials")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_optuna, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    trials_validos = [t for t in study.trials if t.value is not None and t.value > 0]
    trials_validos.sort(key=lambda t: -t.value)

    print(f"[OPTUNA] {len(trials_validos)} configs validas")
    return study, trials_validos


# ================================================================
# SECAO 9: RELATORIO
# ================================================================

def exibir_top(trials_validos, n=15):
    nomes = {1: "Base", 2: "Disc/Prem", 3: "London/NY", 5: "Liq.Sweep"}
    print(f"\n{'='*86}")
    print(f"  TOP {min(n,len(trials_validos))} - SMC OPTIMIZER v7 INSTITUCIONAL")
    print("  (HTF 15min | OPEN exec | swing delay | CPCV | DSR | PBO | Block Bootstrap)")
    print(f"{'='*86}")
    print(
        f"  {'#':>2} {'EST':>10} {'SW':>3} {'RR':>4} {'ATR':>4} "
        f"{'PF':>6} {'DSR':>6} {'WR%':>6} {'Trades':>7} {'DD%':>6} "
        f"{'Jan+':>5} {'Score':>7}"
    )
    print(f"  {'-'*84}")
    for i, t in enumerate(trials_validos[:n], 1):
        p = t.params
        ua = t.user_attrs
        star = "*" if i == 1 else " "
        nome = nomes.get(p.get("estrategia", 1), "?")
        print(
            f"  {star}{i:>2} {nome:>10} "
            f"{p.get('swing_length',0):>3} "
            f"{p.get('rr_min',0):>4} "
            f"{p.get('atr_mult_sl',0):>4} "
            f"{ua.get('profit_factor',0):>6.3f} "
            f"{ua.get('dsr',0):>6.3f} "
            f"{ua.get('win_rate',0):>6.1f} "
            f"{ua.get('total_trades',0):>7} "
            f"{ua.get('max_drawdown',0):>6.1f} "
            f"{ua.get('janelas_pos',0):>5.0f}% "
            f"{t.value:>7.4f}"
        )
    print(f"{'='*86}")


def relatorio_final(m, mc, stress, sq, cpcv_taxa, pbo_val, pbo_interp, titulo="RESULTADO", config=None):
    if not m:
        return

    nomes = {1: "Base", 2: "Disc/Prem", 3: "London/NY", 5: "Liq.Sweep"}
    sep = "=" * 66

    def L(lb, vl):
        print(f"  {lb:<36} {str(vl):>26}")

    print(f"\n{sep}")
    print(f"  SMC OPTIMIZER v7 INSTITUCIONAL - {titulo}")
    print(sep)

    if config:
        print(f"  Estrategia : {nomes.get(config.get('estrategia',1),'?')}")
        print(
            f"  Params     : SW={config.get('swing_length')} "
            f"RR={config.get('rr_min')} ATR={config.get('atr_mult_sl')}"
        )
        print("  HTF Bias   : 15min obrigatorio")
        print("  Entrada    : OPEN proximo candle")
        print(f"  Swing Delay: {config.get('swing_length')} candles")
        print(f"  {'-'*62}")

    L("Total Trades", m["total_trades"])
    L("Wins / Losses", f"{m['wins']} W  /  {m['losses']} L")
    L("Win Rate", f"{m['win_rate']}%")
    L("Profit Factor", m["profit_factor"])
    L("Sharpe Ratio", m["sharpe_ratio"])
    L("Sortino Ratio", m["sortino_ratio"])
    L("Calmar Ratio", m["calmar_ratio"])
    L("DSR (Deflated Sharpe)", f"{m['dsr']} - {m['dsr_interp']}")
    L("Expectancy", f"R$ {m['expectancy_brl']:,.2f}")
    L("Total PnL", f"R$ {m['total_pnl_brl']:,.2f}")
    L("Retorno %", f"{m['retorno_pct']}%")
    L("Max Drawdown", f"{m['max_drawdown_pct']}%")
    L("Capital Final", f"R$ {m['capital_final']:,.2f}")
    L("Janelas Positivas", f"{m['pct_janelas_positivas']}%")
    L("Skewness PnL", m["skewness_pnl"])
    L("Kurtosis PnL", m["kurtosis_pnl"])

    print(f"  {'-'*62}")
    print("  VALIDACOES ESTATISTICAS")
    L("PBO (overfitting risk)", f"{pbo_val} - {pbo_interp}")
    cpcv_status = "APROVADO" if cpcv_taxa >= 0.60 else "REPROVADO"
    L("CPCV paths lucrativos", f"{cpcv_taxa*100:.0f}% ({cpcv_status})")

    if stress:
        L("Stress - Sem top 5", "OK" if stress.get("sem_top5_lucrativo") else "FALHOU")
        L("Stress - Slippage 2x", "OK" if stress.get("slippage_2x_lucrativo") else "FALHOU")
        L("Stress - Comissao +50%", "OK" if stress.get("comissao_50pct_lucrativo") else "FALHOU")
        L("Stress Score", stress.get("stress_score", "?"))

    if sq:
        L("Signal Quality WR", f"{sq.get('signal_quality_wr',0)}%")
        L("Sinais gerados", sq.get("sinais_gerados", 0))

    if mc:
        print(f"  {'-'*62}")
        print(
            f"  BLOCK BOOTSTRAP ({mc.get('n_simulacoes',0):,} sims | {mc.get('metodo','')})"
        )
        L("Prob. Lucro", f"{mc.get('prob_lucro_pct',0)}%")
        L("Retorno Mediana", f"{mc.get('retorno_mediana',0)}%")
        L("Retorno P10/P90", f"{mc.get('retorno_p10',0)}% / {mc.get('retorno_p90',0)}%")
        L("DD Mediano", f"{mc.get('dd_mediano',0)}%")
        L("DD Pior", f"{mc.get('dd_pior',0)}%")
        L("Risco Ruina", f"{mc.get('prob_ruina_pct',0)}%")

    print(sep)

    # Veredicto final
    print("\n  VEREDICTO:")
    checks = [
        m["profit_factor"] > 1.2,
        m["dsr"] > 0.70,
        pbo_val < 0.35,
        cpcv_taxa >= 0.60,
        stress.get("aprovado", False) if stress else False,
        m["pct_janelas_positivas"] >= 60,
    ]
    n_pass = sum(checks)
    labels = ["PF>1.2", "DSR>0.70", "PBO<0.35", "CPCV>=60%", "Stress", "Consistencia"]
    for ok, label in zip(checks, labels):
        print(f"  {'OK' if ok else '--':>4} {label}")
    print(f"\n  {n_pass}/6 criterios aprovados", end=" ")
    if n_pass >= 5:
        print("-> ESTRATEGIA APROVADA PARA PAPER TRADING")
    elif n_pass >= 4:
        print("-> ESTRATEGIA PROMISSORA - mais dados necessarios")
    else:
        print("-> ESTRATEGIA REPROVADA - nao operar")
    print()


# ================================================================
# SECAO 10: MAIN
# ================================================================

def main():
    MINI = "--mini" in sys.argv

    print("=" * 70)
    print("  SMC OPTIMIZER v7 -- NIVEL INSTITUCIONAL")
    print("  HTF 15min | CPCV | DSR | PBO | Block Bootstrap | Optuna")
    print("=" * 70)
    print(
        """
BIBLIOTECAS:

- skfolio: CombinatorialPurgedCV
- arch: CircularBlockBootstrap
- optuna: Bayesian TPE optimization
- scipy: DSR, PBO, estatisticas

CORRECOES CRITICAS:

- Swing lookahead: delay de swing_length candles
- HTF 15min: filtro obrigatorio de bias
- Entrada: OPEN do proximo candle
- Minimo 300 trades (validade estatistica)
"""
    )

    df = carregar()
    split = int(len(df) * 0.70)
    df_ins = df.iloc[:split]
    df_oos = df.iloc[split:]

    print(f"  In-sample : {len(df_ins):,} | {df_ins.index[0].date()} -> {df_ins.index[-1].date()}")
    print(f"  Out-sample: {len(df_oos):,} | {df_oos.index[0].date()} -> {df_oos.index[-1].date()}")

    # Pre-computar indicadores
    preparar_cache(df_ins)

    if MINI:
        print("\n[MINI] Testando 4 configs basicas...")
        configs_mini = [
            (2.0, 1.0, 5, 1),
            (2.5, 1.0, 5, 2),
            (2.0, 0.7, 7, 3),
            (2.0, 1.0, 5, 5),
        ]
        nomes = {1: "Base", 2: "Disc/Prem", 3: "London/NY", 5: "Liq.Sweep"}

        for rr, am, sw, est in configs_mini:
            key = (sw, True)
            df_ind = _DF_IND_CACHE.get(key)
            if df_ind is None:
                continue

            trades, equity = backtest(
                df_ind,
                rr_min=rr,
                atr_mult_sl=am,
                poi_janela=20,
                choch_janela=30,
                estrategia=est,
            )
            m = metricas(trades, equity, n_trials=4)
            nome = nomes.get(est, "?")
            if m:
                dsr_str = f"DSR={m['dsr']}"
                print(
                    f"  {nome:>12}: PF={m['profit_factor']:.3f} | "
                    f"WR={m['win_rate']:.1f}% | Trades={m['total_trades']} | "
                    f"{dsr_str}"
                )
            else:
                print(f"  {nome:>12}: < {MIN_TRADES} trades")

        # Signal Quality no mini
        key_sq = (5, True)
        if key_sq in _DF_IND_CACHE:
            print("\n[SQ] Signal Quality Analysis...")
            sq = signal_quality(_DF_IND_CACHE[key_sq])
            print(f"  Sinais gerados: {sq['sinais_gerados']} | SQ WR: {sq['signal_quality_wr']}%")
        return

    # === OTIMIZACAO BAYESIANA (Optuna) ===
    study, trials_validos = otimizar_optuna(n_trials=N_OPTUNA_TRIALS)

    if not trials_validos:
        print(f"\n[INFO] Nenhuma config gerou >= {MIN_TRADES} trades.")
        print("Com HTF + swing delay, a estrategia e mais seletiva.")
        print("Considere dados de periodo maior.")
        return

    exibir_top(trials_validos)

    # Melhor config
    melhor_trial = trials_validos[0]
    p = melhor_trial.params
    CONFIG = {
        "swing_length": p["swing_length"],
        "rr_min": p["rr_min"],
        "atr_mult_sl": p["atr_mult_sl"],
        "estrategia": p["estrategia"],
        "poi_janela": 20,
        "choch_janela": 30,
        "close_break": True,
    }

    # Coletar retornos de todas as configs para PBO
    print("\n[PBO] Calculando Probability of Backtest Overfitting...")
    retornos_configs = {}
    for t in trials_validos[:20]:
        key_t = (t.params["swing_length"], True)
        df_ind_t = _DF_IND_CACHE.get(key_t)
        if df_ind_t is None:
            continue

        trades_t, _ = backtest(
            df_ind_t,
            rr_min=t.params["rr_min"],
            atr_mult_sl=t.params["atr_mult_sl"],
            poi_janela=20,
            choch_janela=30,
            estrategia=t.params["estrategia"],
        )
        pnls_t = [tr["pnl_brl"] for tr in trades_t if tr.get("resultado") in ("WIN", "LOSS")]
        if len(pnls_t) >= MIN_TRADES:
            cid = f"{t.params['estrategia']}*{t.params['rr_min']}_{t.params['swing_length']}"
            retornos_configs[cid] = np.array(pnls_t)

    pbo_val, pbo_interp = calcular_pbo(retornos_configs)
    print(f"[PBO] {pbo_val} -> {pbo_interp}")

    # CPCV validation
    print("\n[CPCV] Combinatorial Purged Cross-Validation...")
    cpcv_resultados, retornos_oos_cpcv, cpcv_taxa = validar_cpcv(
        df_ins,
        CONFIG,
        n_folds=6,
        n_test_folds=2,
        purge_size=30,
        embargo_size=10,
    )

    # Signal Quality
    print("\n[SQ] Signal Quality Analysis...")
    key_sq = (CONFIG["swing_length"], True)
    sq = signal_quality(_DF_IND_CACHE[key_sq], estrategia=CONFIG["estrategia"])
    print(
        f"[SQ] Sinais:{sq['sinais_gerados']} | "
        f"TP:{sq['sinais_tp']} | SL:{sq['sinais_sl']} | "
        f"WR:{sq['signal_quality_wr']}%"
    )

    # Out-of-sample
    print("\n[OOS] Backtest Out-of-Sample...")
    df_oos_ind = preparar_smc_htf(df_oos, swing_length=CONFIG["swing_length"])
    bt_params = {
        k: v
        for k, v in CONFIG.items()
        if k in ["rr_min", "atr_mult_sl", "poi_janela", "choch_janela", "estrategia"]
    }
    t_oos, e_oos = backtest(df_oos_ind, **bt_params)
    m_oos = metricas(t_oos, e_oos, n_trials=N_OPTUNA_TRIALS)

    # Verificar degradacao
    is_pf = melhor_trial.user_attrs.get("profit_factor", 1)
    oos_pf = m_oos["profit_factor"] if m_oos else 0
    if is_pf > 0 and oos_pf > 0:
        degradacao = (is_pf - oos_pf) / is_pf * 100
        print(f"  Degradacao OOS: {degradacao:.1f}% ({'OK' if degradacao < 40 else 'ATENCAO >40%'})")

    # Full dataset
    print("\n[FULL] Backtest dataset completo...")
    df_full_ind = preparar_smc_htf(df, swing_length=CONFIG["swing_length"])
    t_full, e_full = backtest(df_full_ind, **bt_params)
    m_full = metricas(t_full, e_full, n_trials=N_OPTUNA_TRIALS)

    # Block Bootstrap Monte Carlo
    mc = monte_carlo_block_bootstrap(t_full, n_sim=2000)

    # Stress tests
    print("\n[STRESS] Stress Tests...")
    stress = stress_test(t_full, e_full)
    print(f"  Score: {stress.get('stress_score','?')} | Aprovado: {stress.get('aprovado', False)}")

    # Relatorio final
    relatorio_final(
        m=m_full,
        mc=mc,
        stress=stress,
        sq=sq,
        cpcv_taxa=cpcv_taxa,
        pbo_val=pbo_val,
        pbo_interp=pbo_interp,
        titulo="COMPLETO",
        config=CONFIG,
    )

    # Exportar
    top20_data = []
    for t in trials_validos[:20]:
        top20_data.append(
            {
                "params": t.params,
                "score": t.value,
                **t.user_attrs,
            }
        )

    out = {
        "versao": "v7",
        "gerado_em": datetime.now().isoformat(),
        "config_melhor": CONFIG,
        "metricas_full": m_full,
        "metricas_oos": m_oos,
        "monte_carlo": mc,
        "stress_test": stress,
        "signal_quality": sq,
        "cpcv": {
            "resultados": cpcv_resultados,
            "taxa_sucesso": cpcv_taxa,
        },
        "pbo": {
            "valor": pbo_val,
            "interpretacao": pbo_interp,
        },
        "optuna_top20": top20_data,
        "grid_stats": {
            "n_trials": N_OPTUNA_TRIALS,
            "validos": len(trials_validos),
            "min_trades": MIN_TRADES,
            "max_pf_filtro": MAX_PF,
            "max_sharpe_filtro": MAX_SHARPE,
        },
        "trades": t_full,
        "equity_curve": e_full,
    }
    path = f"{OUTPUT_DIR}/resultado_v7.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[OK] Salvo em {path}")

    nomes = {1: "Base", 2: "Disc/Prem", 3: "London/NY", 5: "Liq.Sweep"}
    print("\nCONCLUIDO!")
    print(f"  Estrategia : {nomes.get(CONFIG['estrategia'], '?')}")
    if m_full:
        print(
            f"  PF={m_full['profit_factor']} | "
            f"DSR={m_full['dsr']} | "
            f"WR={m_full['win_rate']}% | "
            f"Trades={m_full['total_trades']}"
        )
    print(f"  PBO={pbo_val} ({pbo_interp})")
    print(f"  CPCV={cpcv_taxa*100:.0f}% lucrativos")


if __name__ == "__main__":
    main()