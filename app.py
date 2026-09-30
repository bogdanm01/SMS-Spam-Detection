from pathlib import Path

import streamlit as st

from src.model_io import load_final_model
from src.predict import predict_sms

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_model.pt"


@st.cache_resource(show_spinner="Učitavanje modela...")
def get_model():
    return load_final_model(MODEL_PATH)


def main() -> None:
    st.set_page_config(
        page_title="SMS Spam Detection",
        page_icon="✉️",
        layout="centered",
    )

    st.title("SMS Spam Detection")
    st.write("Proveri da li model novu SMS poruku prepoznaje kao regularnu ili neželjenu.")

    if not MODEL_PATH.is_file():
        st.error("Sačuvani model nije pronađen na models/final_model.pt.")
        st.write("Ako pokrećeš aplikaciju lokalno, napravi checkpoint iz korena projekta:")
        st.code("python -m src.train_final", language="bash")
        st.write("Ako koristiš Docker, proveri da li je checkpoint kopiran u sliku tokom izgradnje.")
        st.stop()

    with st.form("sms-form"):
        message = st.text_area(
            "Tekst SMS poruke",
            placeholder="Na primer: Claim your free prize now!",
            height=160,
        )
        submitted = st.form_submit_button("Proveri poruku", type="primary", use_container_width=True)

    if submitted:
        if not message.strip():
            st.warning("Unesi SMS poruku pre provere.")
        else:
            model, checkpoint = get_model()
            result = predict_sms(message, model, checkpoint)
            is_spam = result["label"] == "spam"

            st.subheader("Rezultat")
            if is_spam:
                st.error("Spam · model je označio poruku kao neželjenu.")
            else:
                st.success("Ham · model je označio poruku kao regularnu.")

            ham_column, spam_column = st.columns(2)
            ham_column.metric("Ham skor", f"{result['ham_score']:.1%}")
            spam_column.metric("Spam skor", f"{result['spam_score']:.1%}")
            st.caption(
                "Skorovi su softmax izlaz modela. Njihov zbir je 100%, "
                "ali nisu provereno kalibrisane verovatnoće."
            )

    with st.expander("Kako model dolazi do odluke?"):
        st.write(
            "Tekst se pretvara u tokene i ID-jeve pomoću rečnika sačuvanog uz model. "
            "Transformer obrađuje tokene u kontekstu poruke, a klasifikacioni sloj "
            "vraća dva skora: za ham i spam. Tokom ove provere model ne trenira."
        )
        st.caption("Na izdvojenom test skupu F1 za spam iznosi 0,8983.")


if __name__ == "__main__":
    main()
