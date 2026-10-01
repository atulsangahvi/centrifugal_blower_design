
from __future__ import annotations

import hmac
import streamlit as st

SESSION_KEY = "app_authenticated"
PASSWORD_WIDGET_KEY = "global_app_password"


def get_configured_password() -> str:
    try:
        if "APP_PASSWORD" in st.secrets:
            return str(st.secrets["APP_PASSWORD"])
        if "auth" in st.secrets and "password" in st.secrets["auth"]:
            return str(st.secrets["auth"]["password"])
    except Exception:
        return ""
    return ""


def is_authenticated() -> bool:
    configured = get_configured_password()
    return True if not configured else bool(st.session_state.get(SESSION_KEY, False))


def require_password() -> None:
    configured = get_configured_password()
    if not configured:
        return

    if st.session_state.get(SESSION_KEY, False):
        with st.sidebar:
            st.success("Authenticated")
            if st.button("Log out", key="global_logout_button"):
                st.session_state[SESSION_KEY] = False
                st.session_state.pop(PASSWORD_WIDGET_KEY, None)
                st.rerun()
        return

    with st.sidebar:
        st.divider()
        st.subheader("Login")
        entered = st.text_input(
            "Password",
            type="password",
            key=PASSWORD_WIDGET_KEY,
            placeholder="Enter app password",
        )
        if entered:
            if hmac.compare_digest(str(entered), configured):
                st.session_state[SESSION_KEY] = True
                st.session_state.pop(PASSWORD_WIDGET_KEY, None)
                st.rerun()
            else:
                st.error("Incorrect password")

    st.warning("Enter the app password in the sidebar to continue.")
    st.stop()
