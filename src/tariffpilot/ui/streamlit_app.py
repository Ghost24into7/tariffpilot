"""Small Streamlit UI. Run: streamlit run src/tariffpilot/ui/streamlit_app.py  (pip install '.[ui]')"""
import streamlit as st

from tariffpilot.service import build_service


@st.cache_resource
def svc():
    return build_service()


st.set_page_config(page_title="TariffPilot", page_icon="📦")
st.title("TariffPilot")
st.caption("Decision support for HTS classification. Public data only. Not legal advice.")
s = svc()
text = st.text_area("Describe the product (material, function, use)", height=120)
extra = st.text_input("Extra details (optional)")
if st.button("Classify", type="primary") and text.strip():
    steps = st.empty()
    log: list[str] = []

    def on_event(e):
        log.append(f"{e['type']} {e.get('tool', '')} {e.get('text', '')}".strip())
        steps.code("\n".join(log))

    r = s.classify(text, extra, on_event)
    st.subheader(f"Status: {r.status}")
    if r.question:
        st.info(r.question)
    if r.hts_code:
        st.metric("HTS code", r.hts_code, f"confidence {r.confidence:.0%}")
        if r.duty_estimate:
            st.write(f"General duty rate: **{r.duty_estimate.rate_text}**")
        st.write("Citations:", [f"{c.type}:{c.id}" for c in r.citations])
        st.write("Alternatives:", [a.code for a in r.alternatives])
    if r.restrictions:
        st.warning(" ".join(r.restrictions))
    if r.abstain_reason:
        st.error(r.abstain_reason)
    st.caption(f"trace_id: {r.trace_id} | {r.disclaimer}")
    with st.expander("Guardrail events"):
        st.json([e.__dict__ for e in r.guard_events])
