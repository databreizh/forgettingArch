# cfg/real_flows/real_flows.py

from __future__ import annotations
from typing import List, Iterable
import random

from cfg.core.graph import CollaborativeForgettingGraph
from cfg.core.nodes import Node, RecomputabilityProfile


def _mk_node(
    node_id: str,
    node_type: str,
    owners: Iterable[str] | None = None,
    recomputable: bool = True,
    recompute_cost: int | None = 1,
    deletion_cost: int = 1,
    reason: str = "",
) -> Node:
    """
    Helper factory to create Node objects for synthetic real-world flows.
    
    Encapsulates the creation of RecomputabilityProfile and the Node itself 
    to simplify graph construction.
    """
    rp = RecomputabilityProfile(
        recomputable=recomputable,
        recompute_cost=recompute_cost,
        reason=reason,
    )

    return Node(
        node_id=node_id,
        node_type=node_type,
        owner=set(owners or []),
        recomputability=rp,
        deletion_cost=deletion_cost,
    )


# --------------------------------------------------------------------
# Real-flow 1 : Mutualized Crowdsourcing Pipeline
# --------------------------------------------------------------------

def build_crowd_flow(seed: int = 0) -> CollaborativeForgettingGraph:
    """
    Generates a pseudo-realistic CFG for a mutualized crowdsourcing and ML workflow.

    Workflow Structure:
        workers -> annotations -> aggregation -> QC -> composite labels -> models -> outputs
    
    This flow simulates a platform where many workers provide data that is 
    aggregated and then consumed by different organizations (OrgA, OrgB).
    """
    rnd = random.Random(seed)
    g = CollaborativeForgettingGraph()

    # --- Workers and annotations ------------------------------------
    num_workers = 40
    ann_ids: List[str] = []

    for i in range(num_workers):
        wid = f"w{i}"
        ann_id = f"ann_{i}"
        
        # Raw contribution: Source data, cannot be recomputed if lost.
        w_node = _mk_node(
            node_id=wid,
            node_type="input",
            owners=[f"worker_{i}"],
            deletion_cost=1,
            recomputable=False,
            recompute_cost=None,
            reason="raw_contribution",
        )
        g.add_node(w_node)

        # Stored annotation: Derived from a worker, typically one-to-one strong link.
        ann_node = _mk_node(
            node_id=ann_id,
            node_type="annotation",
            owners=[f"worker_{i}", "platform"],
            deletion_cost=2,
            recomputable=False,
            recompute_cost=None,
            reason="stored_annotation",
        )
        g.add_node(ann_node)
        g.add_dependency(wid, ann_id, dep_type="strong")
        ann_ids.append(ann_id)

    # --- Aggregation nodes (majority vote) --------------------------
    num_aggs = 8
    agg_ids: List[str] = []

    for j in range(num_aggs):
        aid = f"agg_{j}"
        a_node = _mk_node(
            node_id=aid,
            node_type="aggregation",
            owners=["platform"],
            deletion_cost=5,
            recomputable=True,
            recompute_cost=3,
            reason="majority_vote",
        )
        g.add_node(a_node)
        agg_ids.append(aid)

        # Robust aggregation: marked as 'weak' because the loss of a single 
        # annotation might not invalidate the consensus (majority vote logic).
        parents = rnd.sample(ann_ids, k=min(10, len(ann_ids)))
        for p in parents:
            g.add_dependency(p, aid, dep_type="weak")

    # --- Quality-check nodes ----------------------------------------
    qc_ids: List[str] = []
    for j, aid in enumerate(agg_ids):
        qid = f"qc_{j}"
        q_node = _mk_node(
            node_id=qid,
            node_type="qc",
            owners=["platform"],
            deletion_cost=8,
            recomputable=True,
            recompute_cost=6,
            reason="qc_score",
        )
        g.add_node(q_node)
        g.add_dependency(aid, qid, dep_type="strong")
        qc_ids.append(qid)

    # --- Composite labels shared between OrgA / OrgB ----------------
    comp_ids: List[str] = []
    num_comp = 20
    for k in range(num_comp):
        cid = f"comp_{k}"
        c_node = _mk_node(
            node_id=cid,
            node_type="composite_label",
            owners=["OrgA", "OrgB", "platform"],
            deletion_cost=10,
            recomputable=True,
            recompute_cost=7,
            reason="shared_label",
        )
        g.add_node(c_node)
        comp_ids.append(cid)

        parents = rnd.sample(qc_ids, k=min(4, len(qc_ids)))
        for p in parents:
            g.add_dependency(p, cid, dep_type="weak")

    # --- Downstream models (Expensive training, organizational silos) ----
    model_ids: List[str] = []
    
    # OrgA Models: Non-recomputable (e.g., due to lost legacy environment)
    for m_index in range(4):
        mid = f"model_A_{m_index}"
        m_node = _mk_node(
            node_id=mid,
            node_type="model",
            owners=["OrgA"],
            deletion_cost=50,
            recomputable=False,
            recompute_cost=None,
            reason="expensive_legacy_training",
        )
        g.add_node(m_node)
        model_ids.append(mid)

        parents = rnd.sample(comp_ids, k=min(8, len(comp_ids)))
        for p in parents:
            g.add_dependency(p, mid, dep_type="strong")

    # OrgB Models: Recomputable but expensive
    for m_index in range(3):
        mid = f"model_B_{m_index}"
        m_node = _mk_node(
            node_id=mid,
            node_type="model",
            owners=["OrgB"],
            deletion_cost=40,
            recomputable=True,
            recompute_cost=30,
            reason="retrainable_model",
        )
        g.add_node(m_node)
        model_ids.append(mid)

        parents = rnd.sample(comp_ids, k=min(6, len(comp_ids)))
        for p in parents:
            g.add_dependency(p, mid, dep_type="strong")

    # --- Outputs consumed by OrgC -----------------------------------
    for idx, mid in enumerate(model_ids):
        oid = f"out_{idx}"
        o_node = _mk_node(
            node_id=oid,
            node_type="output",
            owners=["OrgC"],
            deletion_cost=5,
            recomputable=True,
            recompute_cost=2,
            reason="cached_predictions",
        )
        g.add_node(o_node)
        g.add_dependency(mid, oid, dep_type="strong")

    return g


# --------------------------------------------------------------------
# Real-flow 2 : Hybrid ML Pipeline
# --------------------------------------------------------------------

def build_ml_flow(seed: int = 0) -> CollaborativeForgettingGraph:
    """
    Pseudo-realistic ML workflow:
        raw -> preprocessing -> features -> shared embeddings -> models -> reports
    
    Focuses on data engineering stages where early deletions propagate 
    through high fan-out shared components like embeddings.
    """
    rnd = random.Random(seed)
    g = CollaborativeForgettingGraph()

    # --- Raw data shards ---------------------------------------------
    raw_ids: List[str] = []
    for i in range(20):
        rid = f"raw_{i}"
        r_node = _mk_node(
            node_id=rid,
            node_type="raw",
            owners=["data_team"],
            deletion_cost=3,
            recomputable=False,
            recompute_cost=None,
            reason="source_data",
        )
        g.add_node(r_node)
        raw_ids.append(rid)

    # --- Preprocessing ------------------------------------------------
    prep_ids: List[str] = []
    for rid in raw_ids:
        pid = f"prep_{rid}"
        p_node = _mk_node(
            node_id=pid,
            node_type="preproc",
            owners=["data_team"],
            deletion_cost=4,
            recomputable=True,
            recompute_cost=2,
            reason="preprocessing_logic",
        )
        g.add_node(p_node)
        g.add_dependency(rid, pid, dep_type="strong")
        prep_ids.append(pid)

    # --- Feature extraction -------------------------------------------
    feat_ids: List[str] = []
    for pid in prep_ids:
        fid = f"feat_{pid}"
        f_node = _mk_node(
            node_id=fid,
            node_type="features",
            owners=["data_team"],
            deletion_cost=6,
            recomputable=True,
            recompute_cost=3,
            reason="feature_engineering",
        )
        g.add_node(f_node)
        g.add_dependency(pid, fid, dep_type="strong")
        feat_ids.append(fid)

    # --- Shared embeddings (High fan-out, high recompute cost) -------
    emb_ids: List[str] = []
    for i in range(5):
        eid = f"emb_{i}"
        e_node = _mk_node(
            node_id=eid,
            node_type="embedding",
            owners=["platform", "OrgA", "OrgB"],
            deletion_cost=30,
            recomputable=True,
            recompute_cost=20,
            reason="shared_embedding",
        )
        g.add_node(e_node)
        emb_ids.append(eid)

        parents = rnd.sample(feat_ids, k=min(10, len(feat_ids)))
        for p in parents:
            g.add_dependency(p, eid, dep_type="strong")

    # --- Downstream models (OrgA & OrgB) -----------------------------
    model_ids: List[str] = []
    
    # OrgA: Standard models
    for m_idx in range(3):
        mid = f"ml_model_A_{m_idx}"
        m_node = _mk_node(
            node_id=mid,
            node_type="model",
            owners=["OrgA"],
            deletion_cost=60,
            recomputable=True,
            recompute_cost=40,
            reason="A_training_cycle",
        )
        g.add_node(m_node)
        model_ids.append(mid)

        parents = rnd.sample(emb_ids, k=min(3, len(emb_ids)))
        for p in parents:
            g.add_dependency(p, mid, dep_type="strong")

    # OrgB: Legacy models (Contractual or technical inability to retrain)
    for m_idx in range(3):
        mid = f"ml_model_B_{m_idx}"
        m_node = _mk_node(
            node_id=mid,
            node_type="model",
            owners=["OrgB"],
            deletion_cost=80,
            recomputable=False,
            recompute_cost=None,
            reason="B_legacy_contractual_constraint",
        )
        g.add_node(m_node)
        model_ids.append(mid)

        parents = rnd.sample(emb_ids, k=min(4, len(emb_ids)))
        for p in parents:
            g.add_dependency(p, mid, dep_type="strong")

    # --- Reports / Analytics -----------------------------------------
    for idx, mid in enumerate(model_ids):
        rid = f"report_{idx}"
        r_node = _mk_node(
            node_id=rid,
            node_type="report",
            owners=["OrgC"],
            deletion_cost=10,
            recomputable=True,
            recompute_cost=5,
            reason="analytics_report",
        )
        g.add_node(r_node)
        g.add_dependency(mid, rid, dep_type="strong")

    return g