import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
from datetime import datetime

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Simulador CIEP: Actuarial", layout="wide")

# --- MODELO DE MORTALIDAD (Proyecto Tablas de mortalidad 2025) ---
q_hombres = np.array([
    0.00526052722195369, 0.000377573574683949, 0.000368785906623848, 0.000349219544180083, 0.00032097758038413,
    0.0002864776373774, 0.000248224179348885, 0.000209452375547418, 0.000173203215617787, 0.000142865537594185,
    0.000122123983771038, 0.000115036445489258, 0.00012639724416585, 0.000161878479835624, 0.000227983925239216,
    0.000332011890345976, 0.000419920578677227, 0.000500579761848741, 0.000569016893557277, 0.000623707100048523,
    0.000690464898647695, 0.000734519364072467, 0.000761077957405564, 0.000775238988840247, 0.000781527498991398,
    0.000783774436229987, 0.00078518431002837, 0.000788148899673906, 0.000794255311046242, 0.000804368195738244,
    0.000818757558695088, 0.00083727794723047, 0.000859548996068777, 0.000885123831364279, 0.000913645507432358,
    0.000944813071756375, 0.000978241535878419, 0.00101345155213057, 0.00105012386611217, 0.0010883768497645,
    0.00112879248227974, 0.00117225203328422, 0.00121981835386321, 0.0012727563462712, 0.00133253043009447,
    0.00140075666262585, 0.00147912608567992, 0.00156938858967635, 0.00167332126474187, 0.00179267918792184,
    0.00192909002000212, 0.0020839709040396, 0.00225845721755158, 0.00245340457903159, 0.00266949412967369,
    0.00290748795374294, 0.00316857219593588, 0.00345476565817071, 0.00376932395933277, 0.00411710272322959,
    0.00450471248793166, 0.00494053081706002, 0.0054345892992021, 0.00599840546862686, 0.00664460901574639,
    0.00738657117171082, 0.00823788411590906, 0.00921162956411433, 0.0103195973878692, 0.0115721788670448,
    0.0129787751725082, 0.0145478333390977, 0.01628674242491, 0.0182041903117874, 0.0203138520384644,
    0.0226340368108806, 0.0251846622457092, 0.0279862401738191, 0.0310654950783106, 0.0344600296300744,
    0.038212282556454, 0.0423657933478447, 0.046968917356797, 0.0520644915985513, 0.0576886691485409,
    0.0638799797429742, 0.0706781404270348, 0.0781373664002667, 0.0863201922313814, 0.0952927105406334,
    0.105149184428404, 0.116459614666869, 0.128899238019294, 0.141600993118813, 0.155602640327382,
    0.17043770538163, 0.186048550933076, 0.202376921792886, 0.219365973890509, 0.236961448141553,
    0.255112281980367, 0.27377086360971, 0.292893066627437, 0.312438154961571, 0.332368615687682,
    0.352649955925322, 0.371543160533318, 0.390545632533398, 0.409628615351423, 0.428766003505835,
    1.0
])

q_mujeres = np.array([
    0.00344333551425604, 0.000273198588001602, 0.000250999530470955, 0.000221131476464794, 0.000189630599219363,
    0.000160017625500219, 0.000134786580057289, 0.000115135612772363, 0.000101164245302515, 0.0000924966545438692,
    0.0000883146475879009, 0.0000877948019023276, 0.0000901840995072122, 0.000095440557485772, 0.000104098265766327,
    0.000118072772866149, 0.00012986967077202, 0.000142402142631857, 0.000155187688391133, 0.000168102351519782,
    0.000181280066019723, 0.000195059241598977, 0.000209545803957693, 0.000224859893318381, 0.000241102830314702,
    0.000258196621812207, 0.000278128247193566, 0.000291184867524017, 0.000299626656947697, 0.000306524219806197,
    0.000315103675730647, 0.000328158243947916, 0.000347592328179671, 0.00037401957068745, 0.000406820920774933,
    0.000444405172418786, 0.000484536935835483, 0.000524864817247044, 0.000563619566768679, 0.000600264490865971,
    0.000635709962705061, 0.000671878081493946, 0.00071090504003529, 0.000754376518952622, 0.000802932908143207,
    0.000856463520174053, 0.000914798623910806, 0.000978542101292987, 0.00104948651767755, 0.00113033403881313,
    0.00122373241023181, 0.00133107097538146, 0.00145157541442257, 0.00158217668890827, 0.00171832539839655,
    0.00185576395005447, 0.00199282995571883, 0.00213254029052324, 0.00228359115135575, 0.00245985201005196,
    0.0026783545952187, 0.00295632021740655, 0.00330803226829946, 0.00374232154706333, 0.00426124746423318,
    0.00486037031654884, 0.0055308479835416, 0.00626250612197264, 0.00704738120636447, 0.0078832963026803,
    0.00877599486250189, 0.00973688281044129, 0.0107804291276072, 0.0119270515586315, 0.0132036921624903,
    0.0146370498228139, 0.0162543872642357, 0.018088724266634, 0.0201782727804631, 0.0225717277595187,
    0.0253266955371419, 0.0285018625181639, 0.0321583983651666, 0.0363483762615505, 0.0411157771703588,
    0.0465035412397594, 0.0525437319924597, 0.0592714160702584, 0.0667214489155665, 0.0749040068337621,
    0.0838374952662448, 0.0940143783428071, 0.104975091504375, 0.116664239960081, 0.128997472054893,
    0.142225865390459, 0.156370048202817, 0.171373993525494, 0.187252477085999, 0.204020413304947,
    0.221692856036161, 0.240284999308558, 0.259812178069839, 0.28028986893197, 0.301733690918448,
    0.324159406213365, 0.345335222153168, 0.367228247107341, 0.389838481075885, 0.4131659240588,
    1.0
])

def calcular_vpa_interno(sexo, tasa, tipo_renta, frec_nombre, años_t, jubilacion, enfoque):
    v = 1 / (1 + tasa)
    q_x = q_hombres if sexo == "Masculino" else q_mujeres
    p_x = 1 - q_x
    map_frec = {"Mensual": 12, "Bimestral": 6, "Trimestral": 4, "Anual": 1}
    n_pagos = map_frec[frec_nombre]
    
    # 1. Factor de Supervivencia 0 a Jubilación
    # Si es "Financiero Puro", la probabilidad es 1 (100%)
    prob_0_jub = np.prod(p_x[0:jubilacion]) if enfoque == "Actuarial-Actuarial" else 1.0
    
    # 2. Factor de Anualidad
    factor_anualidad = 0
    if tipo_renta == "Vitalicia":
        t_p_jub = 1.0
        for t in range(jubilacion, 111):
            factor_anualidad += t_p_jub * (v**(t-jubilacion))
            if t < 110: t_p_jub *= p_x[t]
    else:
        for t in range(años_t):
            factor_anualidad += (v**t)
            
    # 3. Capital Semilla (Valor Presente al Año 0)
    # Capital = (Renta_Anual * Anualidad) * Prob_Llegar_Vivo * Descuento_Financiero
    factor_vpa = factor_anualidad * prob_0_jub * (v**jubilacion)
    
    return factor_vpa, n_pagos, factor_anualidad, prob_0_jub

# --- INTERFAZ STREAMLIT ---
st.title("🏦 CIEP - Simulador Actuarial de Capital Semilla")
st.markdown("Capital Semilla")

with st.sidebar:
    st.header("Configuración")
    # --- EL SWITCH SOLICITADO ---
    enfoque_seleccionado = st.selectbox("Enfoque de Cálculo:", ["Actuarial-Actuarial", "Financiero-Actuarial"], index=1)
    modo = st.radio("¿Qué deseas calcular?", ["Capital Semilla", "Renta"])
    sexo = st.selectbox("Sexo:", ["Masculino", "Femenino"])
    edad_jubilacion = st.number_input("Edad de Jubilación:", min_value=1, max_value=100, value=65)
    modo_tasa = st.radio(
        "Modo de ingreso de tasa",
        ["Usar tasa estándar", "Ingresar tasa"],
        index=0  # default: Usar tasa estándar
    )
    if modo_tasa == "Usar tasa estándar":
        tasas = {
            "Conservador (3%)": 0.03,
            "Base (4%)": 0.04,
            "Optimista (5%)": 0.05
        }
        opcion_tasa = st.selectbox(
            "Escenario de tasa de rendimiento",
            options=list(tasas.keys()),
            index=2  # default: Optimista (5%)
        )
        tasa_anual = tasas[opcion_tasa]
    else:
        tasa_anual = st.number_input(
            "Tasa de Rendimiento Anual (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,   # default: 5%
            step=0.1,
            format="%.2f"
        ) / 100
    tipo_renta = st.selectbox("Tipo de Renta:", ["Vitalicia", "Temporal (Periódica)"], index=0)
    años_t = st.number_input("Duración de renta (solo si es Temporal):", min_value=1, value=20)
    frecuencia = st.selectbox("Frecuencia de Renta:", ["Mensual", "Bimestral", "Trimestral", "Anual"], index=1)

# Cálculos principales
f_vpa, n_pagos, f_anualidad_jub, prob_llegada = calcular_vpa_interno(
    sexo, tasa_anual, tipo_renta, frecuencia, años_t, edad_jubilacion, enfoque_seleccionado
)

if modo == "Capital Semilla":
    monto_renta = st.number_input(f"Monto de la Renta {frecuencia} (S/):", value=350.0)
    cap_semilla = (monto_renta * n_pagos) * f_vpa
else:
    cap_semilla = st.number_input("Capital Semilla, Año 0 (S/):", value=1000.0)
    monto_renta = (cap_semilla / f_vpa) / n_pagos

fondo_en_jubilacion = (monto_renta * n_pagos) * f_anualidad_jub

# --- MOSTRAR RESULTADOS ---
col_res1, col_res2, col_res3 = st.columns(3)
col_res1.metric("Capital Semilla (Año 0)", f"S/ {cap_semilla:,.2f}")
col_res2.metric(f"Renta {frecuencia}", f"S/ {monto_renta:,.2f}")
col_res3.metric("Prob. Supervivencia (0 a Jub)", f"{prob_llegada:.2%}" if enfoque_seleccionado == "Actuarial-Actuarial" else "100.00%")

# --- GENERADOR DE EXCEL ---
def generar_excel_con_formulas(sex, tasa, freq, n_pagos, rent, jub, enfoque):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        ws = workbook.add_worksheet('Memoria de Cálculo')
        
        h_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
        m_fmt = workbook.add_format({'num_format': '"S/" #,##0.00', 'border': 1})
        p_fmt = workbook.add_format({'num_format': '0.0000%', 'border': 1})
        
        # 1. Supuestos
        ws.write('A1', 'SUPUESTOS', h_fmt)
        ws.write('A2', 'Renta Periódica'); ws.write('B2', rent, m_fmt)
        ws.write('A3', 'Frecuencia (pagos/año)'); ws.write('B3', n_pagos)
        ws.write('A4', 'Tasa Anual'); ws.write('B4', tasa, p_fmt)
        ws.write('A5', 'Edad Jubilación'); ws.write('B5', jub)
        ws.write('A6', 'Renta Anual Eq.'); ws.write_formula('B6', '=B2*B3', m_fmt)
        ws.write('A7', 'Factor v'); ws.write_formula('B7', '=1/(1+B4)', p_fmt)
        ws.write('A8', 'Enfoque Seleccionado'); ws.write('B8', enfoque)
        
        # 2. Resultados Actuariales Dinámicos
        row_jub = 12 + jub
        ws.write('D1', 'CÁLCULOS CLAVE', h_fmt)
        
        # B11: Probabilidad de supervivencia (condicionada al switch)
        ws.write('D2', 'B11: Factor Supervivencia (0-Jub)')
        if enfoque == "Actuarial-Actuarial":
            ws.write_formula('E2', f'=PRODUCTO(B12:B{row_jub-1})', p_fmt)
        else:
            ws.write('E2', 1.0, p_fmt)
            
        # B10: Valor de la anualidad a los 65
        ws.write('D3', 'B10: Valor Anualidad en Jub')
        ws.write_formula('E3', f'=SUMA(F{row_jub}:F122)*B6', m_fmt)
        
        # Resultado Final
        ws.write('D5', 'CAPITAL SEMILLA AÑO 0', h_fmt)
        ws.write_formula('E5', '=E3*E2*(B7^B5)', m_fmt)

        # 3. Tabla de Mortalidad
        headers = ['Edad', 'px', f'tpx (desde {jub})', 'VP Financiero', 'Pago', 'Suma Actuarial']
        for c, h in enumerate(headers): ws.write(10, c, h, h_fmt)

        q_x_target = q_hombres if sex == "Masculino" else q_mujeres
        for i, edad in enumerate(range(111)):
            r = i + 11
            ws.write(r, 0, edad)
            ws.write(r, 1, 1 - q_x_target[i], p_fmt)
            
            # tpx dinámico según edad de jubilación
            if edad < jub: 
                ws.write(r, 2, 0)
            elif edad == jub: 
                ws.write(r, 2, 1)
            else: 
                ws.write_formula(r, 2, f'=C{r}*B{r}', p_fmt)
            
            # VP y Suma dinámicos
            if edad >= jub:
                ws.write_formula(r, 3, f'=$B$7^(A{r+1}-$B$5)', p_fmt)
                ws.write(r, 4, 1)
                ws.write_formula(r, 5, f'=C{r+1}*D{r+1}', m_fmt)
            else:
                ws.write(r, 3, 0); ws.write(r, 4, 0); ws.write(r, 5, 0)

    return output.getvalue()

# --- BOTÓN DE DESCARGA ---
excel_data = generar_excel_con_formulas(sexo, tasa_anual, frecuencia, n_pagos, monto_renta, edad_jubilacion, enfoque_seleccionado)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

st.download_button(
    label=f"📥 Descargar Cálculo ({enfoque_seleccionado})",
    data=excel_data,
    file_name=f"CIEP_{enfoque_seleccionado}_{timestamp}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Gráfico
st.subheader("Crecimiento del Fondo hasta la Jubilación")
edades_graf = np.arange(0, edad_jubilacion + 1)
saldos_graf = cap_semilla * (1 + tasa_anual)**edades_graf
st.line_chart(pd.DataFrame({"Saldo": saldos_graf}, index=edades_graf))
