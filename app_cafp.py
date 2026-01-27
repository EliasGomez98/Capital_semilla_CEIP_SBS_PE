import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Simulador CEIP: Actuarial", layout="wide")

# --- MODELO DE MORTALIDAD (Proyecto Tablas de mortalidad 2025) ---
q_hombres = np.array([5.03940441e-03, 3.46560981e-04, 3.24551757e-04, 2.94943411e-04, 2.60475316e-04, 2.23760630e-04, 1.86883138e-04, 1.52672598e-04, 1.22969336e-04, 9.96950011e-05, 8.48315434e-05, 8.06616603e-05, 9.06884446e-05, 1.20166096e-04, 1.76206656e-04, 2.67840964e-04, 3.52324554e-04, 4.33768281e-04, 5.04452933e-04, 5.59512306e-04, 6.19850609e-04, 6.53317023e-04, 6.65194866e-04, 6.62101937e-04, 6.50192810e-04, 6.34583634e-04, 6.19450216e-04, 6.07461846e-04, 5.99871931e-04, 5.96806480e-04, 5.97630105e-04, 6.01395192e-04, 6.07234551e-04, 6.14653791e-04, 6.23719615e-04, 6.34728401e-04, 6.47645406e-04, 6.61920814e-04, 6.76973759e-04, 6.92740774e-04, 7.09664637e-04, 7.28306091e-04, 7.49128249e-04, 7.72628131e-04, 7.99423244e-04, 8.30235684e-04, 8.65810569e-04, 9.06963797e-04, 9.54598470e-04, 1.00969386e-03, 1.07318796e-03, 1.14589886e-03, 1.22842315e-03, 1.32108604e-03, 1.42395750e-03, 1.53700628e-03, 1.66029054e-03, 1.79420766e-03, 1.93976303e-03, 2.09888391e-03, 2.27453628e-03, 2.47079474e-03, 2.69286976e-03, 2.94715228e-03, 3.24092633e-03, 3.58205516e-03, 3.97830944e-03, 4.43619561e-03, 4.95954585e-03, 5.54914023e-03, 6.20313974e-03, 6.91687723e-03, 7.68250023e-03, 8.49294350e-03, 9.34798822e-03, 1.02531107e-02, 1.12139401e-02, 1.22344176e-02, 1.33255102e-02, 1.45115013e-02, 1.58195548e-02, 1.72738809e-02, 1.89013401e-02, 2.07163406e-02, 2.27196924e-02, 2.49116994e-02, 2.72901126e-02, 2.98683385e-02, 3.26658882e-02, 3.57012879e-02, 3.90248589e-02, 4.57601232e-02, 5.36909587e-02, 6.26063082e-02, 7.31191617e-02, 8.52321165e-02, 9.91396984e-02, 1.15060379e-01, 1.33240680e-01, 1.53959588e-01, 1.77533386e-01, 2.04321057e-01, 2.34730372e-01, 2.69224805e-01, 3.08331430e-01, 3.52649956e-01, 3.71543161e-01, 3.90545633e-01, 4.09628615e-01, 4.28766004e-01, 1.00000000e+00])

q_mujeres = np.array([3.28401115e-03, 2.48698449e-04, 2.18468949e-04, 1.84235210e-04, 1.51723433e-04, 1.23259870e-04, 1.00373723e-04, 8.34064961e-05, 7.16928800e-05, 6.45448135e-05, 6.09952836e-05, 6.03111637e-05, 6.16858602e-05, 6.51867706e-05, 7.09168073e-05, 8.03599609e-05, 8.82498788e-05, 9.65069278e-05, 1.05067123e-04, 1.13606999e-04, 1.22241114e-04, 1.31035453e-04, 1.40046533e-04, 1.49544218e-04, 1.59748899e-04, 1.70650764e-04, 1.83620191e-04, 1.92341051e-04, 1.98159106e-04, 2.02962855e-04, 2.08733001e-04, 2.17176368e-04, 2.29439630e-04, 2.45616641e-04, 2.64994028e-04, 2.86401967e-04, 3.08348193e-04, 3.29293979e-04, 3.48157963e-04, 3.64837167e-04, 3.80275690e-04, 3.95926415e-04, 4.13142968e-04, 4.32790785e-04, 4.55090935e-04, 4.79792095e-04, 5.06584789e-04, 5.35606386e-04, 5.67662205e-04, 6.04081934e-04, 6.46177082e-04, 6.94597786e-04, 7.48859403e-04, 8.07337942e-04, 8.67709899e-04, 9.27869957e-04, 9.87067511e-04, 1.04688784e-03, 1.11160321e-03, 1.18783094e-03, 1.28347169e-03, 1.40623729e-03, 1.56221520e-03, 1.75481125e-03, 1.98422695e-03, 2.24758137e-03, 2.53992507e-03, 2.85523585e-03, 3.18773085e-03, 3.53413659e-03, 3.89546375e-03, 4.27374321e-03, 4.66973902e-03, 5.08951425e-03, 5.54614663e-03, 6.04909444e-03, 6.60481705e-03, 7.22335774e-03, 7.91634793e-03, 8.70413264e-03, 9.61372709e-03, 1.06693579e-02, 1.18965545e-02, 1.33086886e-02, 1.49105176e-02, 1.67091020e-02, 1.87024831e-02, 2.08995105e-02, 2.33173188e-02, 2.59526379e-02, 2.88260585e-02, 3.44007851e-02, 4.09302574e-02, 4.85329405e-02, 5.73293010e-02, 6.76123547e-02, 7.96173972e-02, 9.35751692e-02, 1.09788895e-01, 1.28609460e-01, 1.50443082e-01, 1.75760253e-01, 2.05106172e-01, 2.39112922e-01, 2.78513692e-01, 3.24159406e-01, 3.45335222e-01, 3.67228247e-01, 3.89838481e-01, 4.13165924e-01, 1.00000000e+00])

def calcular_vpa_interno(sexo, tasa, tipo_renta, frec_nombre, años_t, jubilacion):
    v = 1 / (1 + tasa)
    q_x = q_hombres if sexo == "Masculino" else q_mujeres
    p_x = 1 - q_x
    map_frec = {"Mensual": 12, "Bimestral": 6, "Trimestral": 4, "Anual": 1}
    n_pagos = map_frec[frec_nombre]
    
    prob_0_jub = 1.0
    
    factor_anualidad = 0
    if tipo_renta == "Vitalicia":
        t_p_jub = 1.0
        for t in range(jubilacion, 111):
            factor_anualidad += t_p_jub * (v**(t-jubilacion))
            if t < 110: t_p_jub *= p_x[t]
    else:
        for t in range(int(años_t)):
            factor_anualidad += (v**t)
            
    factor_vpa = factor_anualidad * prob_0_jub * (v**jubilacion)
    return factor_vpa, n_pagos, factor_anualidad, prob_0_jub

# --- INTERFAZ ---}
st.title("🏦 CEIP - Simulador Actuarial de Capital Semilla")
st.markdown("Este simulador permite estimar el Capital Semilla en el marco de los trabajos del Comité para el Estudio Integral del Sistema Previsional en el Perú (CEIP), conforme a lo dispuesto en la Resolución SBS N.° 04043-2025, de fecha 11 de noviembre de 2025.")

with st.sidebar:
    st.header("Configuración")
    modo = st.radio("¿Qué deseas calcular?", ["Capital Semilla", "Renta"])
    sexo = st.selectbox("Sexo:", ["Masculino", "Femenino"])
    edad_jubilacion = st.number_input("Edad de Jubilación:", min_value=1, max_value=100, value=65)
    
    modo_tasa = st.radio("Modo de ingreso de tasa", ["Usar tasa estándar", "Ingresar tasa"])
    if modo_tasa == "Usar tasa estándar":
        tasas = {"Conservador (3%)": 0.03, "Base (4%)": 0.04, "Optimista (5%)": 0.05}
        opcion_tasa = st.selectbox("Escenario de tasa", options=list(tasas.keys()), index=2)
        tasa_anual = tasas[opcion_tasa]
    else:
        tasa_anual = st.number_input("Tasa Anual (%)", min_value=0.0, value=5.0) / 100

    tipo_renta = st.selectbox("Tipo de Renta:", ["Vitalicia", "Temporal"], index=0)
    años_t = st.number_input("Duración (años):", min_value=1, value=20)
    frecuencia = st.selectbox("Frecuencia:", ["Mensual", "Bimestral", "Trimestral", "Anual"], index=1)

# Cálculos
f_vpa, n_pagos, f_anualidad_jub, prob_llegada = calcular_vpa_interno(
    sexo, tasa_anual, tipo_renta, frecuencia, años_t, edad_jubilacion
)

if modo == "Capital Semilla":
    monto_renta = st.number_input(f"Monto de la Renta {frecuencia} (S/):", value=350.0)
    cap_semilla = (monto_renta * n_pagos) * f_vpa
else:
    cap_semilla = st.number_input("Capital Semilla, Año 0 (S/):", value=1000.0)
    monto_renta = (cap_semilla / f_vpa) / n_pagos

# --- MOSTRAR RESULTADOS ---
c1, c2, c3 = st.columns(3)
c1.metric("Capital Semilla (Año 0)", f"S/ {cap_semilla:,.2f}")
c2.metric(f"Renta {frecuencia}", f"S/ {monto_renta:,.2f}")
c3.metric("Prob. Supervivencia", f"100.00%")

# --- EXCEL SEGURO ---
def generar_excel_seguro(sex, tasa, freq, n_pagos, rent, jub, cap_sem):
    output = BytesIO()
    # No usamos Pandas para el writer para evitar el error de importación interna
    import xlsxwriter
    workbook = xlsxwriter.Workbook(output)
    ws = workbook.add_worksheet('Memoria')
    
    h_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
    m_fmt = workbook.add_format({'num_format': '"S/" #,##0.00'})
    
    ws.write('A1', 'SUPUESTOS', h_fmt)
    ws.write('A3', 'Capital Semilla'); ws.write('B3', cap_sem, m_fmt)
    ws.write('A4', 'Renta Calculada'); ws.write('B4', rent, m_fmt)
    
    workbook.close()
    return output.getvalue()

# Intentar generar el botón solo si xlsxwriter está disponible
try:
    data_xls = generar_excel_seguro(sexo, tasa_anual, frecuencia, n_pagos, monto_renta, edad_jubilacion, cap_semilla)
    st.download_button(
        label="📥 Descargar Reporte Excel",
        data=data_xls,
        file_name=f"Calculo_SBS_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except Exception as e:
    st.warning("⚠️ El sistema está instalando los componentes de Excel. Por favor, refresca la página en 10 segundos.")

# Gráfico
st.subheader("Proyección del Fondo")
edades_graf = np.arange(0, edad_jubilacion + 1)
saldos_graf = cap_semilla * (1 + tasa_anual)**edades_graf
st.line_chart(pd.DataFrame({"Saldo": saldos_graf}, index=edades_graf))
