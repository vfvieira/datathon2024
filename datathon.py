import numpy as np
import scipy.stats as scs
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def plot_quartis(vlr_investimento_acum_quartis_prop,tipo_quartil,outra_propriedade,tipo_filtro):#  (xxxxx,'tipo_quartil','outra_propriedade','tipo_filtro')
	#[programa][quartil_id]
	plt.figure(figsize=(8,8))
	barWidth = 0.28
	labels = [1,2,3,4]
	
	vals_emendas = [vlr_investimento_acum_quartis_prop['EMENDAS'][1],vlr_investimento_acum_quartis_prop['EMENDAS'][2],vlr_investimento_acum_quartis_prop['EMENDAS'][3],vlr_investimento_acum_quartis_prop['EMENDAS'][4]]
	vals_pac_fin = [vlr_investimento_acum_quartis_prop['PAC FIN'][1],vlr_investimento_acum_quartis_prop['PAC FIN'][2],vlr_investimento_acum_quartis_prop['PAC FIN'][3],vlr_investimento_acum_quartis_prop['PAC FIN'][4]]
	
	
	br1 = labels
	br2 = [x + barWidth for x in br1]
	#br3 = [x + barWidth for x in br2]
	
	pl1 = plt.bar(br1,vals_emendas,color='tab:blue',alpha=0.7,label='EMENDAS',width = barWidth)
	pl2 = plt.bar(br2,vals_pac_fin,color='tab:red',alpha=0.7,label='PAC FIN',width = barWidth)
	
	
	plt.xticks([(r+1) + barWidth/2 for r in range(len(labels))],labels)
	
	if tipo_quartil == 'mortes' or tipo_quartil == 'feridos':
		plt.xlabel('Quartis de variação de %s' % tipo_quartil,fontsize=16)
		plt.ylabel('Valor de investimento acumulado (proporção)',fontsize=16)
	elif tipo_quartil == 'vlr_investimento':
		plt.xlabel('Quartis de investimento acumulado (proporção)',fontsize=16)
		plt.ylabel('Variação de %s' % outra_propriedade,fontsize=16)
	
	
	plt.legend(prop={'size': 12})
	
	
	
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	
	plt.savefig("quartis_quartil_%s_versus_%s_%s.png" % (tipo_quartil,outra_propriedade,tipo_filtro),bbox_inches='tight')
	
	plt.clf()
	plt.close()


def plot_tlcc(tlcc_series,propriedades):
	colors = {}
	colors['EMENDAS'] = 'tab:blue'
	colors['PAC FIN'] = 'tab:red'
	markers = {}
	markers['EMENDAS'] = 'o'
	markers['PAC FIN'] = 's'
	max_concat = []
	min_concat = []
	
	plt.figure(figsize=(8,8))
	for programa in tlcc_series[propriedades]:
		max_programa = max(tlcc_series[propriedades][programa])
		max_concat.append(max_programa)
				
		min_programa = min(tlcc_series[propriedades][programa])
		min_concat.append(min_programa)
				
		horizontal = [0 for i in range(len(tlcc_series[propriedades][programa])) ]
		vertical = [min(min_concat)-0.02,max(max_concat)+0.02]
		vert_zero = [0,0]

		offset = [i for i in range(len(tlcc_series[propriedades][programa]))]

		

		plt.plot(offset,tlcc_series[propriedades][programa],color = colors[programa],label = '%s' % programa, linewidth=3, alpha=0.7, marker=markers[programa])
		plt.plot(offset,horizontal,color = 'gray',linewidth=0.8, alpha=0.6)
		
		plt.legend(fontsize=12)
		plt.xlabel("Atraso (anos)",fontsize=16)
		plt.ylabel("Correlação com atraso de tempo",fontsize=16)
		
		plt.tick_params(axis='both', which='major', labelsize=16)
		plt.tick_params(axis='both', which='minor', labelsize=12)

		plt.xticks(offset)

	plt.savefig("tlcc_%s.png" % propriedades,bbox_inches='tight')


	plt.clf()
	plt.close()

def ccdf(x, xlabel, ylabel, nome_prop,log=1):
	type_plot = 'frequency'
	
	if type_plot == 'frequency':
		x, y = sorted(x,reverse=True), np.arange(len(x))# / len(x)
	elif type_plot == 'probability':
		x, y = sorted(x,reverse=True), np.arange(len(x)) / len(x)
	
	if nome_prop == 'taxa_vlr_investimento':
		label = 'Valor de investimento (per capita)'
	elif nome_prop == 'taxa_pop_beneficiada':
		label = 'População beneficiada (proporção)'
	elif nome_prop == 'taxa_media_mortes':
		label = 'Mortes'
	elif nome_prop == 'taxa_media_feridos':
		label = 'Feridos'
	plt.figure(figsize=(8,8))
	plt.plot(x, y,color = 'tab:orange',linewidth=3, alpha=0.7, label = label)
	plt.legend(fontsize=12)
	#plt.legend(loc="upper right",fontsize=12)
	plt.xlabel(xlabel,fontsize=16)
	plt.ylabel(ylabel,fontsize=16)
	plt.xticks(np.arange(0, max(x), 5))
	
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	
	if log == 1:
		plt.yscale('log')
		plt.xscale('log')
	plt.tight_layout()
	plt.savefig("ccdf_%s.png" % nome_prop,bbox_inches='tight')
	plt.clf()
	plt.close()

def get_quartil(quartis,valor):
	
	for quartil_id,quartil_val in enumerate(quartis):
		if valor <= quartil_val:
			return quartil_id

def get_propriedades(carteira_dict,acidentes_dict,populacao_dict,programa,ano_a,ano_b,propriedade_carteira,propriedade_acidentes):
	vals_carteira = []
	vals_acidentes = []
	for cod_ibge in carteira_dict[programa][ano_a]:
		populacao = populacao_dict[ano_a][cod_ibge]
		vals_carteira.append(sum(carteira_dict[programa][ano_a][cod_ibge][propriedade_carteira]))
		vals_acidentes.append(acidentes_dict[propriedade_acidentes][ano_b][cod_ibge])
			
	#end	
	return vals_carteira,vals_acidentes
	
	
def get_propriedade_acum(carteira_dict,programa,ano,propriedade_carteira):
	vals_carteira_acum = []
	for cod_ibge in carteira_dict[programa][ano]:
		populacao = populacao_dict[ano_a][cod_ibge]
		
		val = 0
		for ano_acum in carteira_dict[programa]:
			if ano_acum <= ano:
				val += sum(carteira_dict[programa][ano][cod_ibge][propriedade_carteira])
		
		vals_carteira_acum.append(val)
	return vals_carteira_acum

	

def plot_correlacao_estatica(programas,anos_ambas_bases,resultados_correlacao_estatica):
		
	resultados = ['pop_beneficiada_X_mortes','vlr_investimento_X_mortes','pop_beneficiada_X_feridos','vlr_investimento_X_feridos','pop_beneficiada_X_todos_acidentes','vlr_investimento_X_todos_acidentes']
	
	resultados_label = ['População beneficiada X mortes', 'Valor de investimento X mortes', 'População beneficiada X feridos', 'Valor de investimento X feridos', 'População beneficiada X vítimas', 'Valor de investimento X vítimas']
	
	markers_list = ['o','s','1','v','P','X','D']
	colors_list = ['tab:blue','tab:red','tab:green','tab:orange','tab:purple','tab:brown','tab:pink']
	
	label_programa = programas
	
	horizontal = [ 0 for i in range(len(resultados_correlacao_estatica['all']['vlr_investimento_X_mortes'].keys())) ]
	
	for id_resultado, resultado in enumerate(resultados):
	
	
		plt.figure(figsize=(8,8))
		
		for id_programa, programa in enumerate(programas):
		
			label = programa
			if programa == 'all':
				label = 'Todos os programas'
		
			#print("Programa:",programa)
			#for ano in resultados_correlacao_estatica[programa]['len']:
			#	print("Ano",ano, ":",resultados_correlacao_estatica[programa]['len'][ano],"registros")
			
			plt.plot(list(resultados_correlacao_estatica[programa][resultado].keys()),list(resultados_correlacao_estatica[programa][resultado].values()),color = colors_list[id_programa], label = label, linewidth=3, alpha=0.7, marker=markers_list[id_programa])
			
		plt.plot(resultados_correlacao_estatica[programa][resultado].keys(),horizontal,color = 'gray',linewidth=0.6, alpha=0.7)
		
		

		plt.legend(fontsize=12)
		plt.xlabel("Ano",fontsize=16)
		plt.ylabel(resultados_label[id_resultado],fontsize=16)
		#plt.xticks(np.arange(0, max(x), 5))
		plt.tick_params(axis='both', which='major', labelsize=16)
		plt.tick_params(axis='both', which='minor', labelsize=12)

		#plt.xticks( resultados_correlacao_estatica['all']['vlr_investimento_X_mortes'].keys() )
		plt.xticks( anos_ambas_bases )

		plt.savefig("static_%s.png" % resultados[id_resultado],bbox_inches='tight')


		plt.clf()
		plt.close()

#end


def plot_series_temporais_investimentos(series_temporais):

	#print(series_temporais)
	
	
	
	markers_list = ['o','s']#,'1','s','P','X','D']
	colors_list = ['tab:blue','tab:red']#,'tab:green','tab:red','tab:purple','tab:brown','tab:pink']
	
	#resultados_label = ['População beneficiada X mortes', 'Valor de investimento X mortes', 'População beneficiada X feridos', 'Valor de investimento X feridos', 'População beneficiada X vítimas', 'Valor de investimento X vítimas']
	
	#resultados = ['taxa_vlr_investimento','taxa_mortes','taxa_feridos']#,'vlr_investimento_X_feridos','pop_beneficiada_X_todos_acidentes','vlr_investimento_X_todos_acidentes']
	
	#programas = series_temporais['taxa_pop_beneficiada']
	#label_programa = series_temporais['taxa_pop_beneficiada']
	
	
	plt.figure(figsize=(8,8))
	plt.legend(fontsize=12)
	plt.xlabel("Ano",fontsize=16)
	plt.ylabel('Valor de investimento (per capita)',fontsize=16)
	plt.xticks([2015,2016,2017,2018,2019])
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.plot(list(series_temporais['taxa_vlr_investimento']['EMENDAS'].keys()),list(series_temporais['taxa_vlr_investimento']['EMENDAS'].values()),color = colors_list[0], linewidth=3, alpha=0.7, marker=markers_list[0], label = 'EMENDAS')
	plt.legend(fontsize=12)
	plt.savefig("series_temporais_vlr_investimento_emendas.png" ,bbox_inches='tight')
	plt.clf()
	plt.close()
	
	
	
	
	plt.figure(figsize=(8,8))
	#plt.legend(fontsize=12)
	plt.xlabel("Ano",fontsize=16)
	plt.ylabel('Valor de investimento (per capita)',fontsize=16)
	plt.xticks([2015,2016,2017,2018,2019])
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.plot(list(series_temporais['taxa_vlr_investimento']['PAC FIN'].keys()),list(series_temporais['taxa_vlr_investimento']['PAC FIN'].values()),color = colors_list[1], linewidth=3, alpha=0.7, marker=markers_list[1], label = 'PAC FIN')
	plt.legend(fontsize=12)
	plt.savefig("series_temporais_vlr_investimento_pac_fin.png" ,bbox_inches='tight')
	plt.clf()
	plt.close()
	
	#end
	
def plot_series_temporais_acidentes(series_temporais):

	#print(series_temporais)
	
	
	
	markers_list = ['X','P']#,'1','s','P','X','D']
	colors_list = ['tab:brown','tab:purple']#,'tab:green','tab:red','tab:purple','tab:brown','tab:pink']
	
	#resultados_label = ['População beneficiada X mortes', 'Valor de investimento X mortes', 'População beneficiada X feridos', 'Valor de investimento X feridos', 'População beneficiada X vítimas', 'Valor de investimento X vítimas']
	
	#resultados = ['taxa_vlr_investimento','taxa_mortes','taxa_feridos']#,'vlr_investimento_X_feridos','pop_beneficiada_X_todos_acidentes','vlr_investimento_X_todos_acidentes']
	
	#programas = series_temporais['taxa_pop_beneficiada']
	#label_programa = series_temporais['taxa_pop_beneficiada']
	
	
	plt.figure(figsize=(8,5))
	
	plt.xlabel("Ano",fontsize=16)
	plt.ylabel('Taxa média (100k habitantes)',fontsize=16)
	plt.xticks([2015,2016,2017,2018,2019])
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	plt.plot(list(series_temporais['taxa_mortes']['EMENDAS'].keys()),list(series_temporais['taxa_mortes']['EMENDAS'].values()),color = colors_list[0], label = 'Mortes', linewidth=3, alpha=0.7, marker=markers_list[0])
	plt.plot(list(series_temporais['taxa_feridos']['EMENDAS'].keys()),list(series_temporais['taxa_feridos']['EMENDAS'].values()),color = colors_list[1], label = 'Feridos', linewidth=3, alpha=0.7, marker=markers_list[1])
	plt.legend(fontsize=12)
	plt.savefig("series_temporais_acidentes.png" ,bbox_inches='tight')
	plt.clf()
	plt.close()
	
	
	
	
	
	
	#end
	
	
def plot_series_temporais(series_temporais):

	print(series_temporais)
	
	
	
	markers_list = ['o','v','1','s','P','X','D']
	colors_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink']
	
	#resultados_label = ['População beneficiada X mortes', 'Valor de investimento X mortes', 'População beneficiada X feridos', 'Valor de investimento X feridos', 'População beneficiada X vítimas', 'Valor de investimento X vítimas']
	
	#resultados = ['taxa_vlr_investimento','taxa_mortes','taxa_feridos']#,'vlr_investimento_X_feridos','pop_beneficiada_X_todos_acidentes','vlr_investimento_X_todos_acidentes']
	
	#programas = series_temporais['taxa_pop_beneficiada']
	#label_programa = series_temporais['taxa_pop_beneficiada']
	
	
	plt.figure(figsize=(8,8))
		
	plt.plot(list(series_temporais['taxa_vlr_investimento']['EMENDAS'].keys()),list(series_temporais['taxa_vlr_investimento']['EMENDAS'].values()),color = colors_list[0], label = 'Valor de investimendo (proporção) EMENDAS', linewidth=1.8, alpha=0.7, marker=markers_list[0])
	plt.plot(list(series_temporais['taxa_vlr_investimento']['PAC FIN'].keys()),list(series_temporais['taxa_vlr_investimento']['PAC FIN'].values()),color = colors_list[1], label = 'Valor de investimendo (proporção) PAC FIN', linewidth=1.8, alpha=0.7, marker=markers_list[1])
	plt.plot(list(series_temporais['taxa_mortes']['EMENDAS'].keys()),list(series_temporais['taxa_mortes']['EMENDAS'].values()),color = colors_list[2], label = 'Taxa de mortes', linewidth=1.8, alpha=0.7, marker=markers_list[2])
	plt.plot(list(series_temporais['taxa_feridos']['EMENDAS'].keys()),list(series_temporais['taxa_feridos']['EMENDAS'].values()),color = colors_list[3], label = 'Taxa de feridos', linewidth=1.8, alpha=0.7, marker=markers_list[3])
			
	plt.legend(fontsize=16)
	plt.xlabel("Ano",fontsize=16)
	##plt.ylabel(resultados_label[id_resultado],fontsize=16)
	plt.xticks([2015,2016,2017,2018,2019])
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)

	#plt.xticks( resultados_correlacao_estatica['all']['vlr_investimento_X_mortes'].keys() )
	####plt.xticks( anos_ambas_bases )

	plt.savefig("series_temporais.png" ,bbox_inches='tight')


	plt.clf()
	plt.close()
	
	
	

#end

def plot_histograma(x,label_x,label_y,nome_prop):
	plt.figure(figsize=(8,8))
	plt.xlim(xmin=min(x), xmax = max(x))
	#print(min(x),max(x))
	
	plt.hist(x, bins=50)
	plt.xlabel(label_x,fontsize=16)
	plt.ylabel(label_y,fontsize=16)
	
	plt.savefig("histograma_%s.png" % nome_prop,bbox_inches='tight')
	plt.clf()
	plt.close()
	#exit()
#end

def read_populacao(anos):
	
	populacao_dict = {}
	for ano in anos:
		#print(ano)
		populacao_dict[ano] = {}
		df_populacao = pd.read_csv('data/populacao/%s.csv' % ano, header='infer')
		
		for index, row in df_populacao.iterrows():
			cod_ibge = row['Código IBGE']
			populacao = row['Populacao']
			populacao_dict[ano][cod_ibge] = int(populacao)
		#end
		#print(populacao_dict)
		
		#exit()
	#end
		
		
	
	return populacao_dict
#end

def crosscorr(datax, datay, lag=0, wrap=False):
	#https://gist.github.com/jcheong0428/7d5759f78145fc0dc979337f82c6ea33
	import pandas as pd
	
	d = {}
	d['datax'] = list(datax)
	d['datay'] = list(datay)
	
	df = pd.DataFrame(d)
	
	datax = df['datax']
	datay = df['datay']
	
		
	""" Lag-N cross correlation. 
	Shifted data filled with NaNs 
	
	Parameters
	----------
	lag : int, default 0
	datax, datay : pandas.Series objects of equal length
	Returns
	----------
	crosscorr : float
	"""
	if wrap:
		shiftedy = datay.shift(lag)
		shiftedy.iloc[:lag] = datay.iloc[-lag:].values
		return datax.corr(shiftedy,method='spearman')
	else: 
		return datax.corr(datay.shift(lag),method='spearman')
		

	



if __name__ == "__main__":
	
	# Leitura da carteira em um DataFrame
	print("\nLendo informações sobre carteira de empreendimentos (base de dados do SIMU)...",end='')
	carteira_filename = "data/simu-carteira-mun-T.csv"
	df_carteira = pd.read_csv(carteira_filename, header='infer')
	
	# Vamos trabalhar apenas com empreendimentos cuja obra já esteja concluída. Assim conseguimos fazer uma análise do seu impacto nos acidentes.
	df_carteira = df_carteira.loc[ df_carteira['situacao_obra'].isin(['CONCLUIDA','OBJETO CONCLUÍDO']) ]
	print("OK!")
	
	
	
	## Tipos de programa:
	# AVANÇAR PÚBLICO: 340 registros na base de dados original
	#		O Programa Avançar Cidades – Mobilidade Urbana tem o objetivo de melhorar a qualidade dos deslocamentos da população nos ambientes urbanos pelo financiamento de ações de mobilidade urbana voltadas ao transporte público coletivo, ao transporte não motorizado (transporte ativo), à elaboração de planos de mobilidade urbana municipais e metropolitanos, estudos e projetos básicos e executivos.
	#		https://www.gov.br/cidades/pt-br/acesso-a-informacao/acoes-e-programas/mobilidade-urbana/avancar-cidades-2013-mobilidade-urbana
	# EMENDAS: 69038 registros na base de dados original
	# PAC FIN: 1519 registros na base de dados original
	#		FIN: Estados, DF, Consórcios, Municípios, Empresas Privadas Concessionárias, Sub-Concessionárias ou Empresas Autorizadas a operar os serviços públicos de saneamento básico.
	# PAC OGU: 221 registros na base de dados original
	#		OGU: Estados, DF, Consórcios e Municípios cujos serviços não estejam concedidos à iniciativa privada.
	# PRÓ COMUNIDADE: 6 registros na base de dados original
	# SETOR PRIVADO: 121 registros na base de dados original
	
	
	print("\nDescobrindo os tipos de programas...",end = '')
	# Descobrindo os tipos de programa na base de dados e sua frequência por ano
	# (aqui são exibidos apenas programas com valor na coluna ano_fim_obra)
	programas = {}
	df_carteira_by_programa_by_ano_fim_obra = df_carteira.groupby(['programa','ano_fim_obra'])
	count = 0
	print("\n")
	for programa_ano_fim_obra in df_carteira_by_programa_by_ano_fim_obra:
		
		programa = programa_ano_fim_obra[0][0]
		ano_fim_obra = int(programa_ano_fim_obra[0][1])
		
		print("Programa",programa,"| Ano",ano_fim_obra,":",len(programa_ano_fim_obra[1]))
		
		programas[programa] = count
		count+=1
	programas['all'] = count
	print("OK!")
	
	# AVANÇAR PÚBLICO (Total: # 214)
	#Ano 2020 : 6 # 6	// Ano não contemplado na base de dados de acidentes
	#Ano 2021 : 41 # 44	// Ano não contemplado na base de dados de acidentes
	#Ano 2022 : 43 # 65	// Ano não contemplado na base de dados de acidentes
	#Ano 2023 : 13 # 99	// Ano não contemplado na base de dados de acidentes
	# <<NÃO VAMOS>> CONSIDERAR O AVANÇAR PÚBLICO
	
	#EMENDAS (Total: # 45504)
	#Ano 2015 : 22916 # 23579
	#Ano 2016 : 1111 # 1131
	#Ano 2017 : 1748 # 1764
	#Ano 2018 : 2398 # 2423
	#Ano 2019 : 3046 # 3052
	#Ano 2020 : 3467 # 3477	// Ano não contemplado na base de dados de acidentes
	#Ano 2021 : 3533 # 3537	// Ano não contemplado na base de dados de acidentes
	#Ano 2022 : 2943 # 4979	// Ano não contemplado na base de dados de acidentes
	#Ano 2023 : 958 # 1560	// Ano não contemplado na base de dados de acidentes
	# <<VAMOS>> CONSIDERAR O EMENDAS NOS ANOS COM REGISTROS NA BASE DE ACIDENTES
	
	#PAC FIN (Total: 654)
	#Ano 2012 : 5 # 5
	#Ano 2013 : 13 # 13
	#Ano 2014 : 32 # 32
	#Ano 2015 : 37 # 37
	#Ano 2016 : 50 # 51
	#Ano 2017 : 66 # 68
	#Ano 2018 : 53 # 57
	#Ano 2019 : 103 # 105
	#Ano 2020 : 88 # 99	// Ano não contemplado na base de dados de acidentes
	#Ano 2021 : 46 # 52	// Ano não contemplado na base de dados de acidentes
	#Ano 2022 : 36 # 60	// Ano não contemplado na base de dados de acidentes
	#Ano 2023 : 3 # 75	// Ano não contemplado na base de dados de acidentes
	# <<VAMOS>> CONSIDERAR O PAC FIN NOS ANOS COM REGISTROS NA BASE DE ACIDENTES
	
	#PAC OGU (Total: 54)
	#Ano 2015 : 2 # 3
	#Ano 2016 : 1 # 2
	#Ano 2017 : 4 # 4
	#Ano 2018 : 4 # 6
	#Ano 2019 : 4 # 6
	#Ano 2020 : 6 # 7	// Ano não contemplado na base de dados de acidentes
	#Ano 2021 : 5 # 12	// Ano não contemplado na base de dados de acidentes
	#Ano 2022 : 3 # 5	// Ano não contemplado na base de dados de acidentes
	#Ano 2023 : 0 # 9	// Ano não contemplado na base de dados de acidentes
	
	#Programa PRÓ COMUNIDADE (Total: 5)
	#Ano 2000 : 2 # 2	// Ano não contemplado na base de dados de acidentes
	#Ano 2002 : 1 # 1	// Ano não contemplado na base de dados de acidentes
	#Ano 2003 : 1 # 1	// Ano não contemplado na base de dados de acidentes
	#Ano 2017 : 0 # 1
	# <<NÃO VAMOS>> CONSIDERAR O PRÓ COMUNIDADE
	
	#SETOR PRIVADO (Total: 113)
	#Ano 2008 : 1 # 1	// Ano não contemplado na base de dados de acidentes
	#Ano 2012 : 1 # 1	
	#Ano 2017 : 1 # 1
	#Ano 2018 : 51 # 51
	#Ano 2019 : 42 # 42
	#Ano 2020 : 7 # 7	// Ano não contemplado na base de dados de acidentes
	#Ano 2022 : 10 # 10	// Ano não contemplado na base de dados de acidentes
	# <<VAMOS>> CONSIDERAR O SETOR PRIVADO NOS ANOS COM REGISTROS NA BASE DE ACIDENTES
	
		
	# Vamos trabalhar apenas com anos com sobreposição entre as bases de Carteira de Empreendimentos e de Acidentes
	# (Os anos de 2012 e 2013 têm registros, mas poucos. Por isso optou-se por removê-los da análise)
	
	
	# Vamos ler a população da base de dados do IBGE, pois as informações sobre população, tanto na base de dados
	#   da Carteira de Empreendimentos, como na base de dados de acidentes do Atlas da Violência, apresentam
	#   muitos valores com erros.
	#   (obs.: até o desenvolvimento dessa etapa eu não tinha percebido que havia bases de dados complementares no site do Datathon 2024)
	anos_para_analise = [2015,2016,2017,2018,2019]
	print("\nLendo a população dos municípios (base de dados do IBGE)...", end='')
	populacao_dict = read_populacao(anos_para_analise)
	print("OK!")
	# OK, leu a população da base de dados do IBGE.
	
	print("\nDescobrindo o volume de investimento por ano em cada município...",end = '')
	# Descobrindo o volume de investimento por ano em cada município
	vlr_investimento_total = {}
	df_carteira_by_programa_by_ano_fim_obra_by_codigo_ibge = df_carteira.groupby(['programa','ano_fim_obra','Código IBGE'])
	count = 0
	for programa_ano_fim_obra_codigo_ibge in df_carteira_by_programa_by_ano_fim_obra_by_codigo_ibge:
		programa = programa_ano_fim_obra_codigo_ibge[0][0]
		#print(programa_ano_fim_obra[0][1])
		ano_fim_obra = int(programa_ano_fim_obra_codigo_ibge[0][1])
		codigo_ibge = int(programa_ano_fim_obra_codigo_ibge[0][2])
		if codigo_ibge != 0:
			try:
				populacao = populacao_dict[ano_fim_obra][codigo_ibge]
			except KeyError:
				populacao = 1
				if ano_fim_obra in [2015,2016,2017,2018,2019]:
					print("ERRO!!",ano_fim_obra,codigo_ibge)
					exit()
			
			if programa not in vlr_investimento_total:
				vlr_investimento_total[programa] = {}
			if ano_fim_obra not in vlr_investimento_total[programa]:
				vlr_investimento_total[programa][ano_fim_obra] = 0
			
			#print("Programa",programa,"| Ano",ano_fim_obra,":", "| Código IBGE: ", codigo_ibge, len(programa_ano_fim_obra_codigo_ibge[1]))
						
			for empreendimento_id,empreendimento in programa_ano_fim_obra_codigo_ibge[1].iterrows():
				vlr_investimento = empreendimento['vlr_investimento'] / populacao
				vlr_investimento_total[programa][ano_fim_obra] += vlr_investimento
			#end
	#end
			
		
	print("OK!")
	
	
	
	print("\nArmazenando as informações da carteira de empreendimentos...",end='')
	# Percorrendo o df_carteira e armazenando as informações relevantes em um dicionário
	# (essa estrutura vai fazer parte da base de todas as análises)	
	df_carteira_by_ano_fim_obra_cod_ibge = df_carteira.groupby(['ano_fim_obra','Código IBGE'])
	carteira_dict = {}
	for programa in programas:
		carteira_dict[programa] = {}
	
	for ano_fim_obra_cod_ibge in df_carteira_by_ano_fim_obra_cod_ibge:
		ano_fim_obra = int(ano_fim_obra_cod_ibge[0][0])
		cod_ibge = int(ano_fim_obra_cod_ibge[0][1])
		
		for empreendimento_id,empreendimento in ano_fim_obra_cod_ibge[1].iterrows():
			try:			
				programa = empreendimento['programa'] # programa vem da carteira de empreendimentos
				pop_beneficiada = empreendimento['pop_beneficiada'] # pop_beneficiada vem da carteira de empreendimentos
				vlr_investimento = empreendimento['vlr_investimento'] # vlr_investimento vem da carteira de empreendimentos
				populacao = populacao_dict[ano_fim_obra][cod_ibge] # populacao vem da base do IBGE (já lida anteriormente)
				
				taxa_pop_beneficiada = pop_beneficiada / populacao # as taxas serão calculadas com base nas informações da carteira e do IBGE
				taxa_vlr_investimento = vlr_investimento / populacao
				
				
				if ano_fim_obra not in carteira_dict[programa]:
					carteira_dict[programa][ano_fim_obra] = {}
				if ano_fim_obra not in carteira_dict['all']:
					carteira_dict['all'][ano_fim_obra] = {}
				
				if cod_ibge not in carteira_dict[programa][ano_fim_obra]:
					carteira_dict[programa][ano_fim_obra][cod_ibge] = {}
					carteira_dict[programa][ano_fim_obra][cod_ibge]['taxa_pop_beneficiada'] = []
					carteira_dict[programa][ano_fim_obra][cod_ibge]['taxa_vlr_investimento'] = []
					
					carteira_dict[programa][ano_fim_obra][cod_ibge]['pop_beneficiada'] = []
					carteira_dict[programa][ano_fim_obra][cod_ibge]['vlr_investimento'] = []
					
					carteira_dict['all'][ano_fim_obra][cod_ibge] = {}
					carteira_dict['all'][ano_fim_obra][cod_ibge]['taxa_pop_beneficiada'] = []
					carteira_dict['all'][ano_fim_obra][cod_ibge]['taxa_vlr_investimento'] = []
					
					carteira_dict['all'][ano_fim_obra][cod_ibge]['pop_beneficiada'] = []
					carteira_dict['all'][ano_fim_obra][cod_ibge]['vlr_investimento'] = []
					# OK, agora já tenho onde armazenar os valores.
				#end if
				
				carteira_dict[programa][ano_fim_obra][cod_ibge]['taxa_pop_beneficiada'].append(taxa_pop_beneficiada)
				carteira_dict['all'][ano_fim_obra][cod_ibge]['taxa_pop_beneficiada'].append(taxa_pop_beneficiada)
				carteira_dict[programa][ano_fim_obra][cod_ibge]['taxa_vlr_investimento'].append(taxa_vlr_investimento)
				carteira_dict['all'][ano_fim_obra][cod_ibge]['taxa_vlr_investimento'].append(taxa_vlr_investimento)
				
				carteira_dict[programa][ano_fim_obra][cod_ibge]['pop_beneficiada'].append(pop_beneficiada)
				carteira_dict['all'][ano_fim_obra][cod_ibge]['pop_beneficiada'].append(pop_beneficiada)
				carteira_dict[programa][ano_fim_obra][cod_ibge]['vlr_investimento'].append(vlr_investimento)
				carteira_dict['all'][ano_fim_obra][cod_ibge]['vlr_investimento'].append(vlr_investimento)
				
			except KeyError:
				#print('passed\n')
				pass
		
		#end for
	#end for
	print("OK!")
	# OK, leu a carteira de empreendimentos
	
	
	
	print("\nLendo informações sobre acidentes de trânsito (base de dados do Atlas de Violência)...",end='')	
	# Agora vamos armazenar as informações sobre acidentes.
	acidentes_filename = "data/Acidentes de Transportes.csv"
	df_acidentes = pd.read_csv(acidentes_filename, header='infer')
	print("OK!")
	
	
	print("\nArmazenando as informações sobre acidentes de trânsito...",end='')
	# Vamos considerar a taxa no ano e a variação em relação ao ano anterior
	df_acidentes_by_ano_cod_ibge = df_acidentes.groupby(['ano','Código IBGE'])
	acidentes_dict = {}
	
	anos_ambas_bases = []
	
	acidentes_dict['taxa_mun_mortes'] = {}
	acidentes_dict['variacao_taxa_mun_mortes'] = {}
	acidentes_dict['taxa_mun_feridos'] = {}
	acidentes_dict['taxa_total'] = {}
	acidentes_dict['variacao_total'] = {}
	
	acidentes_dict['total_mortes'] = {}
	acidentes_dict['variacao_total_mortes'] = {}
	acidentes_dict['total_feridos'] = {}
	acidentes_dict['variacao_total_feridos'] = {}
	acidentes_dict['total_acidentes'] = {}
	acidentes_dict['variacao_total_acidentes'] = {}
	for ano_cod_ibge in df_acidentes_by_ano_cod_ibge:
		ano = int(ano_cod_ibge[0][0])
		cod_ibge = int(ano_cod_ibge[0][1])
		
		if ano not in anos_ambas_bases:
			anos_ambas_bases.append(ano)
			
			acidentes_dict['taxa_mun_mortes'][ano] = {}
			acidentes_dict['taxa_mun_feridos'][ano] = {}
			acidentes_dict['taxa_total'][ano] = {}
			
			acidentes_dict['total_mortes'][ano] = {}
			acidentes_dict['total_feridos'][ano] = {}
			acidentes_dict['total_acidentes'][ano] = {}
			
			if ano-1 in anos_ambas_bases:
				acidentes_dict['variacao_taxa_mun_mortes'][ano] = {}
				acidentes_dict['variacao_total'][ano] = {}
				
				acidentes_dict['variacao_total_mortes'][ano] = {}
				acidentes_dict['variacao_total_feridos'][ano] = {}
				acidentes_dict['variacao_total_acidentes'][ano] = {}
			
		#end if
		
		for linha_id,linha in ano_cod_ibge[1].iterrows():
			total_mortes = float(linha['total_mortes'])
			total_feridos = float(linha['total_feridos'])
			
			#taxa_mun_mortes = float(linha['taxa_mun_mortes']) # Não posso usar essa coluna da base de dados porque contém muitos valores errados!!
			#taxa_mun_feridos = float(linha['taxa_mun_feridos']) # Não posso usar essa coluna da base de dados porque contém muitos valores errados!!
			
			try:
				populacao = populacao_dict[ano][cod_ibge]
			except KeyError:
				populacao = 1 # Só temos problemas quando o ano é menor que 2011 (e isso acontece fora do intervalo de análise)
			taxa_mun_mortes = (total_mortes / populacao) * 100000 # Vamos pegar a coluna de total_mortes e calcular as mortes por 100k habitantes
			taxa_mun_feridos = (total_feridos / populacao) * 100000 # Vamos pegar a coluna de total_feridos e calcular as mortes por 100k habitantes
			
			
			
			acidentes_dict['taxa_mun_mortes'][ano][cod_ibge] = taxa_mun_mortes
			acidentes_dict['taxa_mun_feridos'][ano][cod_ibge] = taxa_mun_feridos
			acidentes_dict['taxa_total'][ano][cod_ibge] = taxa_mun_mortes + taxa_mun_feridos # Podemos fazer essa sola desse jeito porque o denominador é o mesmo
			
			acidentes_dict['total_mortes'][ano][cod_ibge] = total_mortes
			acidentes_dict['total_feridos'][ano][cod_ibge] = total_feridos
			acidentes_dict['total_acidentes'][ano][cod_ibge] = total_mortes + total_feridos
			
			if ano-1 in anos_ambas_bases:
				acidentes_dict['variacao_taxa_mun_mortes'][ano][cod_ibge] = taxa_mun_mortes - acidentes_dict['taxa_mun_mortes'][ano-1][cod_ibge]
				acidentes_dict['variacao_total'][ano][cod_ibge] = (taxa_mun_mortes + taxa_mun_feridos) - acidentes_dict['taxa_total'][ano-1][cod_ibge]
				
				acidentes_dict['variacao_total_mortes'][ano][cod_ibge] = total_mortes - acidentes_dict['total_mortes'][ano-1][cod_ibge]
				acidentes_dict['variacao_total_feridos'][ano][cod_ibge] = total_feridos - acidentes_dict['total_feridos'][ano-1][cod_ibge]
				acidentes_dict['variacao_total_acidentes'][ano][cod_ibge] = (total_mortes + total_feridos) - acidentes_dict['total_acidentes'][ano-1][cod_ibge]
		#end
		
	#end for ano_cod_ibge
	
	print("OK!")
	
	# OK, leu as informações sobre acidentes.
	
	
	
	
	#"""
	#############################################################################
	#                   INFORMAÇÕES POR MUNICÍPIO
	#############################################################################
	# Vamos reunir as informações por município para poder gerar gráficos explicativos
		
	# Organizando informação relacionada à carteira de empreendimentos
	taxa_vlr_investimento_municipio = {}
	taxa_pop_beneficiada_municipio = {}
	for programa in carteira_dict: # <===== AQUI EU NÃO SEI SE CONSIDERO TODOS OS PROGRAMAS OU APENAS OS 'programas_para_plotar'
		for ano in anos_para_analise:
			try:
				for cod_ibge in carteira_dict[programa][ano]:
					if cod_ibge not in taxa_vlr_investimento_municipio:
						taxa_vlr_investimento_municipio[cod_ibge] = 0
						taxa_pop_beneficiada_municipio[cod_ibge] = 0
					taxa_vlr_investimento_municipio[cod_ibge] += sum(carteira_dict[programa][ano][cod_ibge]['taxa_vlr_investimento']) # Aqui poderia ser taxa_vlr_investimento
					taxa_pop_beneficiada_municipio[cod_ibge] += sum(carteira_dict[programa][ano][cod_ibge]['taxa_pop_beneficiada']) # Aqui poderia ser taxa_pop_beneficiada
			except KeyError:
				pass
		
	# Organizando o rank dos municípios em relação à 'taxa_vlr_investimento'
	municipios_vlr_investimento = []
	for chave, valor in sorted(taxa_vlr_investimento_municipio.items(), key=lambda cv: cv[1], reverse=True):
		municipios_vlr_investimento.append(chave)
	# Organizando o rank dos municípios em relação à 'taxa_pop_beneficiada'
	municipios_pop_beneficiada = []
	for chave, valor in sorted(taxa_pop_beneficiada_municipio.items(), key=lambda cv: cv[1], reverse=True):
		municipios_pop_beneficiada.append(chave)
			
	
	
	# Organizando informação relacionada aos acidentes
	taxa_mortes_municipio = {}
	taxa_feridos_municipio = {}
	for ano in anos_para_analise:
		try:
			for cod_ibge in acidentes_dict['taxa_mun_mortes'][ano]:
				if cod_ibge not in taxa_mortes_municipio:
					taxa_mortes_municipio[cod_ibge] = 0
					taxa_feridos_municipio[cod_ibge] = 0
				taxa_mortes_municipio[cod_ibge] += acidentes_dict['taxa_mun_mortes'][ano][cod_ibge]
				taxa_feridos_municipio[cod_ibge] += acidentes_dict['taxa_mun_feridos'][ano][cod_ibge]
		except KeyError:
			pass
	
	taxa_media_mortes_municipio = taxa_mortes_municipio.copy()
	taxa_media_feridos_municipio = taxa_feridos_municipio.copy()
	for cod_ibge in taxa_mortes_municipio:
		taxa_media_mortes_municipio[cod_ibge] = taxa_mortes_municipio[cod_ibge] / 5
		taxa_media_feridos_municipio[cod_ibge] = taxa_feridos_municipio[cod_ibge] / 5
	
	# Organizando o rank dos municípios em relação à 'taxa_mun_mortes'
	municipios_mortes = []
	for chave, valor in sorted(taxa_mortes_municipio.items(), key=lambda cv: cv[1], reverse=True):
		municipios_mortes.append(chave)
	# Organizando o rank dos municípios em relação à 'taxa_mun_feridos'	
	municipios_feridos = []
	for chave, valor in sorted(taxa_feridos_municipio.items(), key=lambda cv: cv[1], reverse=True):
		municipios_feridos.append(chave)
	municipios_populacao = []
	for chave, valor in sorted(populacao_dict[2019].items(), key=lambda cv: cv[1], reverse=True):
		municipios_populacao.append(chave)
			
	
	#############################################################################
	#                FIM DE INFORMAÇÕES POR MUNICÍPIO
	#############################################################################
	
	
	
	# Algumas análises estatísticas de algumas variáveis
	ccdf(list(taxa_vlr_investimento_municipio.values()), 'Proporção de valor de investimento por empreendimento','Frequência', 'taxa_vlr_investimento', log=1)
	ccdf(list(taxa_pop_beneficiada_municipio.values()), 'Proporção de população beneficiada por empreendimento','Frequência', 'taxa_pop_beneficiada', log=1)
	ccdf(list(taxa_media_mortes_municipio.values()), 'Taxa média de mortes anual por município','Frequência', 'taxa_media_mortes', log=1)
	ccdf(list(taxa_media_feridos_municipio.values()), 'Taxa média de feridos anual por município','Frequência', 'taxa_media_feridos', log=1)
	
	# Inicialmente, optamos por usar as colunas pop_beneficiada e vlr_investimento, mas observamos que os resultados gerados estavam muito parecidos.
	# Então decidimos avaliar a correlação entre ambas colunas na base de dados original e verificamos que há, de fato, uma correlação muito alta entre elas.
	# Por isso, optamos por seguir com as análise utilizando, essencialmente, a coluna vlr_investimento.
	col_pop_beneficiada = df_carteira['pop_beneficiada'].values
	col_vlr_investimento = df_carteira['vlr_investimento'].values
	print("Correlação entre as colunas pop_beneficiada e vlr_investimento:",stats.spearmanr(col_pop_beneficiada,col_vlr_investimento).correlation)
		
	
	
	
	
	
	#############################################################################
	#                      ANÁLISE DE CORRELAÇÃO ESTÁTICA
	#############################################################################
	print("\n\nAnálise 1: correlação estática")
	print("\nCalculando correlação estática...",end='')	
	resultados_correlacao_estatica = {}
	for programa in programas:
		resultados_correlacao_estatica[programa] = {}
		resultados_correlacao_estatica[programa]['pop_beneficiada_X_mortes'] = {}
		resultados_correlacao_estatica[programa]['vlr_investimento_X_mortes'] = {}
		resultados_correlacao_estatica[programa]['pop_beneficiada_X_feridos'] = {}
		resultados_correlacao_estatica[programa]['vlr_investimento_X_feridos'] = {}
		resultados_correlacao_estatica[programa]['pop_beneficiada_X_todos_acidentes'] = {}
		resultados_correlacao_estatica[programa]['vlr_investimento_X_todos_acidentes'] = {}
		
		resultados_correlacao_estatica[programa]['len'] = {}
		
	for programa in carteira_dict: # Para programa
		for ano in carteira_dict[programa]:
			
			if ano in anos_para_analise: # Apenas se o ano estiver coberto no acidentes_dict
				resultados_correlacao_estatica[programa]['pop_beneficiada_X_mortes'][ano] = {}
				resultados_correlacao_estatica[programa]['vlr_investimento_X_mortes'][ano] = {}
				resultados_correlacao_estatica[programa]['pop_beneficiada_X_feridos'][ano] = {}
				resultados_correlacao_estatica[programa]['vlr_investimento_X_feridos'][ano] = {}
				resultados_correlacao_estatica[programa]['pop_beneficiada_X_todos_acidentes'][ano] = {}
				resultados_correlacao_estatica[programa]['vlr_investimento_X_todos_acidentes'][ano] = {}
				resultados_correlacao_estatica[programa]['len'][ano] = {}
				
				pop_beneficiada = []
				vlr_investimento = []
				
				mortes = []
				feridos = []
				todos_acidentes = []
				
				for cod_ibge in carteira_dict[programa][ano]: 
					if cod_ibge in acidentes_dict['taxa_total'][ano]: # Vamos utilizar apenas municípios que estão em ambos datasets
						pop_beneficiada.append( sum(carteira_dict[programa][ano][cod_ibge]['taxa_pop_beneficiada']) )
						vlr_investimento.append( sum(carteira_dict[programa][ano][cod_ibge]['taxa_vlr_investimento']) )
						
						mortes.append( acidentes_dict['taxa_mun_mortes'][ano][cod_ibge] )
						feridos.append( acidentes_dict['taxa_mun_feridos'][ano][cod_ibge] )
						todos_acidentes.append( acidentes_dict['taxa_total'][ano][cod_ibge] )
					#end
				#end
				
				# Agora temos as listas com os valores considerados relevantes para a análise.
				# Vamos calcular as correlações estáticas para o ano corrente e o programa corrente.
				
				res = stats.spearmanr(pop_beneficiada,mortes)
				resultados_correlacao_estatica[programa]['pop_beneficiada_X_mortes'][ano] = res.correlation
				
				res = stats.spearmanr(vlr_investimento,mortes)
				resultados_correlacao_estatica[programa]['vlr_investimento_X_mortes'][ano] = res.correlation
				
				res = stats.spearmanr(pop_beneficiada,feridos)
				resultados_correlacao_estatica[programa]['pop_beneficiada_X_feridos'][ano] = res.correlation
				
				res = stats.spearmanr(vlr_investimento,feridos)
				resultados_correlacao_estatica[programa]['vlr_investimento_X_feridos'][ano] = res.correlation
				
				res = stats.spearmanr(pop_beneficiada,todos_acidentes)
				resultados_correlacao_estatica[programa]['pop_beneficiada_X_todos_acidentes'][ano] = res.correlation
				
				res = stats.spearmanr(vlr_investimento,todos_acidentes)
				resultados_correlacao_estatica[programa]['vlr_investimento_X_todos_acidentes'][ano] = res.correlation
				
				resultados_correlacao_estatica[programa]['len'][ano] = len(vlr_investimento)
				
				
			# end if ano in acidentes_dict
			else:
				print("Programa:",programa)
				print("Ano de",ano,"está na carteira, mas não tem registro de acidentes")
		# end for ano in carteira_dict[programa]
	
	# end for ano
	print("OK!")
	
	# OK, fim da análise estática.
	
	
	# Removendo alguns programas que têm poucos registros da lista de programas a serem plotados.
	# (não trazem contribuição à análise e poluem a investigação)
	programas_para_plotar = programas.copy()
	del programas_para_plotar['AVANÇAR PÚBLICO']
	del programas_para_plotar['PRÓ COMUNIDADE']
	del programas_para_plotar['PAC OGU']
	print("\nPlotando resultado de correlação estática...",end='')	
	
	plot_correlacao_estatica(programas_para_plotar,anos_para_analise,resultados_correlacao_estatica)
	print("OK!")
	#############################################################################
	#                   FIM DA ANÁLISE DE CORRELAÇÃO ESTÁTICA
	#############################################################################
	
	
	
	# Vamos analisar apenas os programas 'EMENDAS' e 'PAC FIN', que possuem uma série histórica consistente
	programas_tlcc = ['EMENDAS', 'PAC FIN']
	
	anos_programas = {}
	anos_programas['EMENDAS'] = [2015,2016,2017,2018,2019]
	anos_programas['PAC FIN'] = [2015,2016,2017,2018,2019] # Os anos de 2012 e 2013 têm registros, mas poucos.
	anos_programas['all'] = [2015,2016,2017,2018,2019]
	
	
	
	
	
	
	
	
	#############################################################################
	#                   ANÁLISE DE CORRELAÇÃO DINÂMICA (TLCC)
	#############################################################################
	
	#EMENDAS: 5 anos (2015/2016/2017/2018/2019)
	#PAC FIN: 5 anos (2015/2016/2017/2018/2019)
	
	# Montando as séries temporais com as propriedades que serão confrontadas
	# Primeiro precisamos montar e inicializar toda a estrutura
	# (que é bem diferente das séries que já temos armazenadas)	
	series_temporais = {}
	series_temporais['taxa_pop_beneficiada'] = {}
	series_temporais['taxa_vlr_investimento'] = {}
	series_temporais['taxa_pop_beneficiada_acum'] = {}
	series_temporais['taxa_vlr_investimento'] = {}
	series_temporais['taxa_vlr_investimento_acum'] = {}
	series_temporais['taxa_mortes'] = {}
	series_temporais['taxa_feridos'] = {}
	
	
	series_temporais['taxa_pop_beneficiada']['EMENDAS'] = {}
	series_temporais['taxa_pop_beneficiada']['PAC FIN'] = {}
	series_temporais['taxa_vlr_investimento']['EMENDAS'] = {}
	series_temporais['taxa_vlr_investimento']['PAC FIN'] = {}
	
	series_temporais['taxa_pop_beneficiada_acum']['EMENDAS'] = {}
	series_temporais['taxa_pop_beneficiada_acum']['PAC FIN'] = {}
	series_temporais['taxa_vlr_investimento_acum']['EMENDAS'] = {}
	series_temporais['taxa_vlr_investimento_acum']['PAC FIN'] = {}
	
	series_temporais['taxa_mortes']['EMENDAS'] = {}
	series_temporais['taxa_mortes']['PAC FIN'] = {}
	series_temporais['taxa_feridos']['EMENDAS'] = {}
	series_temporais['taxa_feridos']['PAC FIN'] = {}
	
	
	# Estrutura criada e inicializada
		
	# Agora vamos preencher
	total_populacao = {}
	total_taxa_pop_beneficiada = {}
	total_taxa_vlr_investimento = {}
	total_mortes = {}
	total_feridos = {}
	
	variacao_feridos = {}
	count = {}
	
	total_populacao = 0
	total_taxa_pop_beneficiada = 0
	total_taxa_vlr_investimento = 0
	total_mortes = 0
	total_feridos = 0
	
	variacao_feridos = 0
	count = 0
		
	# Primeiro vamos percorrer a carteira_dict
	# (informações sobre a carteira de empreendimentos)
	for programa in programas_tlcc:
		#print("Programa: ",programa)
		for ano in carteira_dict[programa]:
			total_populacao = 0
			total_taxa_pop_beneficiada = 0
			total_taxa_vlr_investimento = 0
			total_mortes = 0
			total_feridos = 0
			count = 0
									
			for cod_ibge in carteira_dict[programa][ano]:
				populacao = populacao_dict[ano][cod_ibge]
				total_populacao += populacao
				for valor_pop_beneficiada,valor_vlr_investimento in zip(carteira_dict[programa][ano][cod_ibge]['pop_beneficiada'],carteira_dict[programa][ano][cod_ibge]['vlr_investimento']):
					total_taxa_pop_beneficiada += valor_pop_beneficiada / populacao
					total_taxa_vlr_investimento += valor_vlr_investimento / populacao
					
				#end
				count += 1
				
			#end
			
			# Temos os valores, agora é só armazenar na estrutura
			series_temporais['taxa_pop_beneficiada'][programa][ano] = float(total_taxa_pop_beneficiada)
			series_temporais['taxa_vlr_investimento'][programa][ano] = float(total_taxa_vlr_investimento)
			
			series_temporais['taxa_pop_beneficiada_acum'][programa][ano] = 0
			series_temporais['taxa_vlr_investimento_acum'][programa][ano] = 0
			for ano_acum in carteira_dict[programa]:
				if ano_acum <= ano:
					series_temporais['taxa_pop_beneficiada_acum'][programa][ano] += series_temporais['taxa_pop_beneficiada'][programa][ano_acum]
					series_temporais['taxa_vlr_investimento_acum'][programa][ano] += series_temporais['taxa_vlr_investimento'][programa][ano_acum]
			#end
			#print(count)
		#end
	#end
	
	# Agora vamos percorrer a acidentes_dict
	# (informações sobre acidentes)
	
	tlcc_serie = {}
	tlcc_serie['taxa_vlr_investimento_X_feridos'] = {}
	tlcc_serie['taxa_vlr_investimento_X_mortes'] = {}
	
	for programa in programas_tlcc:
		for ano in carteira_dict[programa]:
			total_populacao = 0
			total_mortes = 0
			total_feridos = 0
			
			variacao_feridos = 0
			count = 0
									
			for cod_ibge in carteira_dict[programa][ano]:
				populacao = populacao_dict[ano][cod_ibge]
				total_populacao += populacao
				
				total_mortes += acidentes_dict['total_mortes'][ano][cod_ibge]
				total_feridos += acidentes_dict['total_feridos'][ano][cod_ibge]
				
				count += 1
				
			#end
			
			# Temos os valores, agora é só armazenar na estrutura
			try:
				series_temporais['taxa_mortes'][programa][ano] = float(total_mortes) / float(total_populacao)
				series_temporais['taxa_feridos'][programa][ano] = float(total_feridos) / float(total_populacao)
				
			except ZeroDivisionError:
				series_temporais['taxa_mortes'][programa][ano] = 0
				series_temporais['taxa_feridos'][programa][ano] = 0
			
		#end
		
		tlcc_serie['taxa_vlr_investimento_X_feridos'][programa] = []
		tlcc_serie['taxa_vlr_investimento_X_mortes'][programa] = []
				
		
		print('\n\n\n%s:'% programa)
		print('\nTLCC pop_beneficiada X feridos:')
		for i in range(0,len(anos_programas[programa])-1,1):
			corr = crosscorr(series_temporais['taxa_feridos'][programa].values(),series_temporais['taxa_pop_beneficiada'][programa].values(),lag=i)
			print(i,'\t',corr)
				
				
		print('TLCC pop_beneficiada X mortes:')
		for i in range(0,len(anos_programas[programa])-1,1):
			corr = crosscorr(series_temporais['taxa_mortes'][programa].values(),series_temporais['taxa_pop_beneficiada'][programa].values(),lag=i)
			print(i,'\t',corr)
				
		print('TLCC: vlr_investimento X feridos:')
		for i in range(0,len(anos_programas[programa])-1,1):
			corr = crosscorr(series_temporais['taxa_feridos'][programa].values(),series_temporais['taxa_vlr_investimento'][programa].values(),lag=i)
			print(i,'\t',corr)
			tlcc_serie['taxa_vlr_investimento_X_feridos'][programa].append(corr)
				
		print('TLCC: vlr_investimento X mortes:')
		for i in range(0,len(anos_programas[programa])-1,1):
			corr = crosscorr(series_temporais['taxa_mortes'][programa].values(),series_temporais['taxa_vlr_investimento'][programa].values(),lag=i)
			print(i,'\t',corr)
			tlcc_serie['taxa_vlr_investimento_X_mortes'][programa].append(corr)
	#end for programa
	
	plot_series_temporais_investimentos(series_temporais)
	plot_series_temporais_acidentes(series_temporais)
	
	plot_tlcc(tlcc_serie,'taxa_vlr_investimento_X_feridos')
	plot_tlcc(tlcc_serie,'taxa_vlr_investimento_X_mortes')
	
	#############################################################################
	#                 FIM DA ANÁLISE DE CORRELAÇÃO DINÂMICA (TLCC)
	#############################################################################
	
	
	
		
	
	
	
	#############################################################################
	#       	ANÁLISE DE UMA PROPRIEDADE SOB A PERSPECTIVA DE OUTRA 
	#############################################################################
	
	
	print("\n\n\n")
	# Primeiro, calculando as propriedades.
	
	taxa_pop_beneficiada_acum = {}
	taxa_vlr_investimento_acum = {}
	for programa in programas_tlcc:
		taxa_pop_beneficiada_acum[programa] = {}
		taxa_vlr_investimento_acum[programa] = {}
	#end
	
	
	for programa in programas_tlcc:
		for ano in anos_programas[programa]:
			for cod_ibge in carteira_dict[programa][ano]:
				if cod_ibge not in taxa_pop_beneficiada_acum[programa]:
					taxa_pop_beneficiada_acum[programa][cod_ibge] = 0
					taxa_vlr_investimento_acum[programa][cod_ibge] = 0
				#end
				taxa_pop_beneficiada_acum[programa][cod_ibge] += sum(carteira_dict[programa][ano][cod_ibge]['taxa_pop_beneficiada'])
				taxa_vlr_investimento_acum[programa][cod_ibge] += sum(carteira_dict[programa][ano][cod_ibge]['taxa_vlr_investimento'])
		#end
	#end
	
	variacao_taxa_mortes = {}
	variacao_taxa_feridos = {}
	
	for programa in programas_tlcc:
		variacao_taxa_mortes[programa] = {}
		variacao_taxa_feridos[programa] = {}
		
	for cod_ibge in acidentes_dict['taxa_mun_mortes'][2019]:
		variacao_taxa_mortes['EMENDAS'][cod_ibge] = acidentes_dict['taxa_mun_mortes'][2019][cod_ibge] - acidentes_dict['taxa_mun_mortes'][2015][cod_ibge]
		variacao_taxa_feridos['EMENDAS'][cod_ibge] = acidentes_dict['taxa_mun_feridos'][2019][cod_ibge] - acidentes_dict['taxa_mun_feridos'][2015][cod_ibge]
		variacao_taxa_mortes['PAC FIN'][cod_ibge] = acidentes_dict['taxa_mun_mortes'][2019][cod_ibge] - acidentes_dict['taxa_mun_mortes'][2015][cod_ibge]
		variacao_taxa_feridos['PAC FIN'][cod_ibge] = acidentes_dict['taxa_mun_feridos'][2019][cod_ibge] - acidentes_dict['taxa_mun_feridos'][2015][cod_ibge]
		
	#end
	
	
	
	# OK, construiu as estruturas de valores acumulados por programa
	
	
	
	###############################################################################
	# Olhando sob a perspectiva vlr_investimento -> mortes/feridos
	###############################################################################
	# Obtendo os quartis de vlr_investimento
	quartis = np.quantile(list(taxa_vlr_investimento_acum['EMENDAS'].values()), [0,0.25,0.5,0.75,1])
	#print(quartis)
	vlr_investimento_acum_quartis_mortes = {}
	vlr_investimento_acum_quartis_feridos = {}
	variacao_taxa_mortes_quartis = {}
	variacao_taxa_feridos_quartis = {}
	for programa in programas_tlcc:
		vlr_investimento_acum_quartis_mortes[programa] = {}
		vlr_investimento_acum_quartis_feridos[programa] = {}
		variacao_taxa_mortes_quartis[programa] = {}
		variacao_taxa_feridos_quartis[programa] = {}
		for quartil_id in range(len(quartis)):
			vlr_investimento_acum_quartis_mortes[programa][quartil_id] = []
			vlr_investimento_acum_quartis_feridos[programa][quartil_id] = []
			variacao_taxa_mortes_quartis[programa][quartil_id] = []
			variacao_taxa_feridos_quartis[programa][quartil_id] = []
		#end
	#end
	
	
	# Avaliando a variação das mortes pela perspectiva do vlr_investimento
	for programa in programas_tlcc:
		for cod_ibge in taxa_vlr_investimento_acum[programa]:
			# Vamos analisar como os quartis se comportam, mas considerando apenas municípios com quedas no número de mortos e feridos, respectivamente
			if variacao_taxa_mortes[programa][cod_ibge] <= 0:
				quartil_id = get_quartil(quartis,taxa_vlr_investimento_acum[programa][cod_ibge])
				
				vlr_investimento_acum_quartis_mortes[programa][quartil_id].append(taxa_vlr_investimento_acum[programa][cod_ibge])
				variacao_taxa_mortes_quartis[programa][quartil_id].append(variacao_taxa_mortes[programa][cod_ibge])
		#end
	#end
	for programa in programas_tlcc:
		for quartil_id in range(len(quartis)):
			try:
				vlr_investimento_acum_quartis_mortes[programa][quartil_id] = sum(vlr_investimento_acum_quartis_mortes[programa][quartil_id]) / len(vlr_investimento_acum_quartis_mortes[programa][quartil_id])
				variacao_taxa_mortes_quartis[programa][quartil_id] = sum(variacao_taxa_mortes_quartis[programa][quartil_id]) / len(variacao_taxa_mortes_quartis[programa][quartil_id])
			except ZeroDivisionError:
				print('Erro no quartil', quartil_id) # Esse erro pode ser devido aos valores nulos
				variacao_taxa_mortes_quartis[programa][quartil_id] = 0
				vlr_investimento_acum_quartis_mortes[programa][quartil_id] = 0
	# Imprimindo os resultados na tela		
	for programa in programas_tlcc:
		print("\nPrograma:",programa)
		for quartil_id in range(len(quartis)):
			print("Quartil:",quartil_id)
			print('Vlr_investimento:',vlr_investimento_acum_quartis_mortes[programa][quartil_id],'| Variação mortes:',variacao_taxa_mortes_quartis[programa][quartil_id])
	
	
	
	# Avaliando a variação dos feridos pela perspectiva do vlr_investimento		
	for programa in programas_tlcc:
		for cod_ibge in taxa_vlr_investimento_acum[programa]:
			if variacao_taxa_feridos[programa][cod_ibge] <= 0:
				quartil_id = get_quartil(quartis,taxa_vlr_investimento_acum[programa][cod_ibge])
				
				vlr_investimento_acum_quartis_feridos[programa][quartil_id].append(taxa_vlr_investimento_acum[programa][cod_ibge])
				variacao_taxa_feridos_quartis[programa][quartil_id].append(variacao_taxa_feridos[programa][cod_ibge])
		#end
	#end
	for programa in programas_tlcc:
		for quartil_id in range(len(quartis)):
			try:
				vlr_investimento_acum_quartis_feridos[programa][quartil_id] = sum(vlr_investimento_acum_quartis_feridos[programa][quartil_id]) / len(vlr_investimento_acum_quartis_feridos[programa][quartil_id])
				variacao_taxa_feridos_quartis[programa][quartil_id] = sum(variacao_taxa_feridos_quartis[programa][quartil_id]) / len(variacao_taxa_feridos_quartis[programa][quartil_id])
			except ZeroDivisionError:
					print('Erro no quartil', quartil_id)
					variacao_taxa_feridos_quartis[programa][quartil_id] = 0
					vlr_investimento_acum_quartis_feridos[programa][quartil_id] = 0
	# Imprimindo os resultados na tela
	for programa in programas_tlcc:
		print("\nPrograma:",programa)
		for quartil_id in range(len(quartis)):
			print("Quartil:",quartil_id)
			print('Vlr_investimento:',vlr_investimento_acum_quartis_feridos[programa][quartil_id],'| Variação feridos:',variacao_taxa_feridos_quartis[programa][quartil_id])
	
	# Plotando os resultados
	plot_quartis(variacao_taxa_feridos_quartis,'vlr_investimento','feridos','municipios_reducao')
	plot_quartis(variacao_taxa_mortes_quartis,'vlr_investimento','mortes','municipios_reducao')
	

	##################################################
	##################################################
	
	
	###############################################################################
	# Olhando sob a perspectiva mortes -> vlr_investimento
	###############################################################################	
	
	
	
	quartis = np.quantile(list(variacao_taxa_mortes['EMENDAS'].values()), [0,0.25,0.5,0.75,1])
	
	vlr_investimento_acum_quartis_mortes = {}
	vlr_investimento_acum_quartis_feridos = {}
	variacao_taxa_mortes_quartis = {}
	variacao_taxa_feridos_quartis = {}
	
	for programa in programas_tlcc:
		variacao_taxa_mortes_quartis[programa] = {}
		vlr_investimento_acum_quartis_mortes[programa] = {}
		
		
		for quartil_id in range(len(quartis)):
			variacao_taxa_mortes_quartis[programa][quartil_id] = []
			vlr_investimento_acum_quartis_mortes[programa][quartil_id] = []
		#end
	#end
	
	
	# Avaliando as mortes dos municípios
	for programa in programas_tlcc:
		for cod_ibge in taxa_vlr_investimento_acum[programa]:
			#if variacao_taxa_mortes[programa][cod_ibge] <= 0:
			quartil_id = get_quartil(quartis,variacao_taxa_mortes['EMENDAS'][cod_ibge])
			#print("entrou aquiiiiii", cod_ibge, quartil_id)
			
			variacao_taxa_mortes_quartis[programa][quartil_id].append(variacao_taxa_mortes[programa][cod_ibge])
			vlr_investimento_acum_quartis_mortes[programa][quartil_id].append(taxa_vlr_investimento_acum[programa][cod_ibge])
		
		#end
	#end
	
	for programa in programas_tlcc:
		for quartil_id in range(len(quartis)):
			try:
				variacao_taxa_mortes_quartis[programa][quartil_id] = sum(variacao_taxa_mortes_quartis[programa][quartil_id]) / len(variacao_taxa_mortes_quartis[programa][quartil_id])
				vlr_investimento_acum_quartis_mortes[programa][quartil_id] = sum(vlr_investimento_acum_quartis_mortes[programa][quartil_id]) / len(vlr_investimento_acum_quartis_mortes[programa][quartil_id])
			except ZeroDivisionError:
				print('Erro no quartil', quartil_id)
				vlr_investimento_acum_quartis_mortes[programa][quartil_id] = 0
				variacao_taxa_mortes_quartis[programa][quartil_id] = 0
	# Imprimindo os resultados na tela		
	for programa in programas_tlcc:
		print("\nPrograma:",programa)
		for quartil_id in range(len(quartis)):
			print("Quartil:",quartil_id)
			print('Variação mortes:',variacao_taxa_mortes_quartis[programa][quartil_id],'| Vlr_investimento:',vlr_investimento_acum_quartis_mortes[programa][quartil_id])#,'| Variação feridos:',variacao_taxa_feridos_quartis[programa][quartil_id])
			
	plot_quartis(vlr_investimento_acum_quartis_mortes,'mortes','vlr_investimento','todos_municipios')	
			
			
	
	
	###############################################################################
	# Olhando sob a perspectiva feridos -> vlr_investimento
	###############################################################################	
			
	quartis = np.quantile(list(variacao_taxa_feridos['EMENDAS'].values()), [0,0.25,0.5,0.75,1])
	print(quartis)
	
	
	vlr_investimento_acum_quartis_feridos = {}
	vlr_investimento_acum_quartis_feridos = {}
	variacao_taxa_feridos_quartis = {}
	variacao_taxa_feridos_quartis = {}
	
	for programa in programas_tlcc:
		variacao_taxa_feridos_quartis[programa] = {}
		vlr_investimento_acum_quartis_feridos[programa] = {}
		
		
		for quartil_id in range(len(quartis)):
			variacao_taxa_feridos_quartis[programa][quartil_id] = []
			vlr_investimento_acum_quartis_feridos[programa][quartil_id] = []
		#end
	#end
	
	for programa in programas_tlcc:
		for cod_ibge in taxa_vlr_investimento_acum[programa]:
			#if variacao_taxa_feridos[programa][cod_ibge] <= 0:
			quartil_id = get_quartil(quartis,variacao_taxa_feridos['EMENDAS'][cod_ibge])
			#print("entrou aquiiiiii", cod_ibge, quartil_id)
			
			variacao_taxa_feridos_quartis[programa][quartil_id].append(variacao_taxa_feridos[programa][cod_ibge])
			vlr_investimento_acum_quartis_feridos[programa][quartil_id].append(taxa_vlr_investimento_acum[programa][cod_ibge])
		
		#end
	#end
	
	for programa in programas_tlcc:
		for quartil_id in range(len(quartis)):
			try:
				variacao_taxa_feridos_quartis[programa][quartil_id] = sum(variacao_taxa_feridos_quartis[programa][quartil_id]) / len(variacao_taxa_feridos_quartis[programa][quartil_id])
				vlr_investimento_acum_quartis_feridos[programa][quartil_id] = sum(vlr_investimento_acum_quartis_feridos[programa][quartil_id]) / len(vlr_investimento_acum_quartis_feridos[programa][quartil_id])
			except ZeroDivisionError:
				print('Erro no quartil', quartil_id)
				vlr_investimento_acum_quartis_feridos[programa][quartil_id] = 0
				variacao_taxa_feridos_quartis[programa][quartil_id] = 0
	# Imprimindo os resultados na tela		
	for programa in programas_tlcc:
		print("\nPrograma:",programa)
		for quartil_id in range(len(quartis)):
			print("Quartil:",quartil_id)
			print('Variação feridos:',variacao_taxa_feridos_quartis[programa][quartil_id],'| Vlr_investimento:',vlr_investimento_acum_quartis_feridos[programa][quartil_id])#,'| Variação feridos:',variacao_taxa_feridos_quartis[programa][quartil_id])
			
	plot_quartis(vlr_investimento_acum_quartis_feridos,'feridos','vlr_investimento','todos_municipios')
				
	
	print("\n\n\n")
	
	#############################################################################
	#		FIM DA ANÁLISE DE UMA PROPRIEDADE SOB A PERSPECTIVA DE OUTRA
	#############################################################################
	
	
	
	#############################################################################
	#                   CIDADES QUE NÃO RECEBERAM INVESTIMENTOS
	#############################################################################
	
	
	municipios_contemplados = []
	municipios_nao_contemplados = []
	municipios_todos = []
	#for empreendimento in df_carteira.iterrows():
	#	cod_ibge = int(empreendimento[1]['Código IBGE'])
	#	if cod_ibge not in municipios_contemplados:
	#		municipios_contemplados.append(cod_ibge)
	col_cod_ibge = df_carteira['Código IBGE'].values		
	print(len(col_cod_ibge), "empreendimentos")
	for val in col_cod_ibge:
		try:
			cod_ibge = int(val)
			if cod_ibge not in municipios_contemplados:
				municipios_contemplados.append(cod_ibge)
		except ValueError:
			pass
	print(len(municipios_contemplados), "municípios contemplados por empreendimentos.")
	#df_acidentes
	
	col_cod_ibge = df_acidentes['Código IBGE'].values
	for val in col_cod_ibge:
		cod_ibge = int(val)
		if cod_ibge not in municipios_todos:
			municipios_todos.append(cod_ibge)
		if cod_ibge not in municipios_contemplados and cod_ibge not in municipios_nao_contemplados:
			municipios_nao_contemplados.append(cod_ibge)
		
	print(len(municipios_nao_contemplados), "municípios NÃO contemplados por empreendimentos.")
	print(len(municipios_todos), "municípios no total.")
	
	#print("Contemplados:", municipios_contemplados)
	#print("Não contemplados:", municipios_nao_contemplados)
	
	#df_carteira_2015.loc[ df_carteira['situacao_obra'].isin(['CONCLUIDA','OBJETO CONCLUÍDO']) ]
	df_acidentes_2015 = df_acidentes.loc[ df_acidentes['ano'] == 2015]
	acidentes_nao_contemplados = {}
	acidentes_nao_contemplados['mortes'] = {}
	acidentes_nao_contemplados['feridos'] = {}
	for acidentes in df_acidentes_2015.iterrows():
		try:
			cod_ibge = int(acidentes[1]['Código IBGE'])
			if cod_ibge in municipios_nao_contemplados:
				mortes = acidentes[1]['total_mortes']
				feridos = acidentes[1]['total_feridos']
				populacao = populacao_dict[2015][cod_ibge]
				
				acidentes_nao_contemplados['mortes'][cod_ibge] = (mortes / populacao) * 100000
				acidentes_nao_contemplados['feridos'][cod_ibge] = (feridos / populacao) * 100000
			#end
		except KeyError:
			print(cod_ibge)
	#end
	
	
	print("\nInvestigando a projeção de mortes evitadas...")
	total_vidas_salvas = 0
	num_municipios = 0
	total_populacao = 0
	total_vlr_investimento = 0
	for cod_ibge_nao_contemplado in municipios_nao_contemplados:
		if acidentes_nao_contemplados['mortes'][cod_ibge_nao_contemplado] > 0.0001:
			num_municipios+=1
			populacao = populacao_dict[2015][cod_ibge_nao_contemplado]
			total_populacao+=populacao
			taxa_mais_proxima = 1000000000000
			municipio_mais_proximo = 0
			for cod_ibge in acidentes_dict['taxa_mun_mortes'][2015]:
				diferenca = abs(acidentes_dict['taxa_mun_mortes'][2015][cod_ibge] - acidentes_nao_contemplados['mortes'][cod_ibge_nao_contemplado])
				if diferenca < taxa_mais_proxima and acidentes_dict['taxa_mun_mortes'][2015][cod_ibge] > 0:#  and variacao_taxa_mortes['PAC FIN'][cod_ibge] < 0:
					# Apenas municípios que tiveram mortes em 2015
					taxa_mais_proxima = acidentes_dict['taxa_mun_mortes'][2015][cod_ibge]
					municipio_mais_proximo = cod_ibge
					vlr_investimento_municipio_mais_proximo = 0
					try:
						vlr_investimento_municipio_mais_proximo += taxa_vlr_investimento_acum['EMENDAS'][cod_ibge]
					except KeyError:
						pass
					try:
						vlr_investimento_municipio_mais_proximo += taxa_vlr_investimento_acum['PAC FIN'][cod_ibge]
					except KeyError:
						pass
					pop_municipio_mais_proximo = populacao_dict[2015][cod_ibge]
					vlr_investimento_convertido = vlr_investimento_municipio_mais_proximo# * populacao
				#end
			#end
			# Temos o município mais próximo
			
			total_vlr_investimento += vlr_investimento_convertido
			#print("Municipio mais próximo:", municipio_mais_proximo)
			#print("Taxa do município mais próximo em 2015:", taxa_mais_proxima)
			#print("Taxa do município mais próximo em 2019:", acidentes_dict['taxa_mun_mortes'][2019][municipio_mais_proximo])
			pop_municipio_mais_proximo = populacao_dict[2015][municipio_mais_proximo]
			#print("População do município mais próximo em 2015:", pop_municipio_mais_proximo)
			variacao_mais_proximo = variacao_taxa_mortes['PAC FIN'][municipio_mais_proximo]
			#print("Variação do município",municipio_mais_proximo,":",variacao_taxa_mortes['PAC FIN'][municipio_mais_proximo])
			
			
			
			
			populacao = populacao_dict[2015][cod_ibge_nao_contemplado]
			#print("População do município não contemplado:",populacao)
			#print("Taxa de acidentes do não contemplado em 2015:", acidentes_nao_contemplados['mortes'][cod_ibge_nao_contemplado])
			#print("Taxa de acidentes do não contemplado em 2019:", acidentes_dict['taxa_mun_mortes'][2019][cod_ibge_nao_contemplado])
			variacao_nao_contemplado = acidentes_dict['taxa_mun_mortes'][2019][cod_ibge_nao_contemplado] - acidentes_dict['taxa_mun_mortes'][2015][cod_ibge_nao_contemplado]
			#print("Variação do município não contemplado:",variacao_nao_contemplado)
			
			# Total do não contemplado em 2019
			total_2019 = (acidentes_dict['taxa_mun_mortes'][2019][cod_ibge_nao_contemplado] * populacao) / 100000
			#print("\nTotal do não contemplado em 2019:", total_2019)
			
			### TÁ TUDO ERRADO AQUI NOS INTERMEDIÁRIOS, REFAZER (NO FIM DEU CERTO, MAS TEM QUE REPENSAR TUDO AQUI)
			
			
			# Usando a projeção
			taxa_2019_projecao = acidentes_dict['taxa_mun_mortes'][2019][cod_ibge_nao_contemplado] + variacao_taxa_mortes['PAC FIN'][municipio_mais_proximo]
			#print("Taxa projetada em 2019:", acidentes_dict['taxa_mun_mortes'][2019][cod_ibge_nao_contemplado], "+", variacao_taxa_mortes['PAC FIN'][municipio_mais_proximo], "=", taxa_2019_projecao)
			total_2019_projecao = (taxa_2019_projecao * populacao) / 100000
			#print("Total projetado em 2019:", total_2019_projecao)
			
			total_salvo_projecao_2019 = total_2019 - total_2019_projecao
			#print("Total salvo em 2019 (projeção):", total_salvo_projecao_2019)
			
			total_vidas_salvas += total_salvo_projecao_2019
			
			
		#end
		
	print("====>Total de mortes evitadas (projeção):", total_vidas_salvas, "em", num_municipios, "municípios (população total =",total_populacao,")")
	print("(a um valor de",total_vlr_investimento/num_municipios,"por pessoa nesses municípios)")
	
	
	
	print("\nInvestigando a projeção de feridos evitados...")
	total_vidas_salvas = 0
	num_municipios = 0
	total_populacao = 0
	total_vlr_investimento = 0
	for cod_ibge_nao_contemplado in municipios_nao_contemplados:
		if acidentes_nao_contemplados['feridos'][cod_ibge_nao_contemplado] > 0.0001:
			num_municipios+=1
			populacao = populacao_dict[2015][cod_ibge_nao_contemplado]
			total_populacao+=populacao
			taxa_mais_proxima = 1000000000000
			municipio_mais_proximo = 0
			for cod_ibge in acidentes_dict['taxa_mun_feridos'][2015]:
				diferenca = abs(acidentes_dict['taxa_mun_feridos'][2015][cod_ibge] - acidentes_nao_contemplados['feridos'][cod_ibge_nao_contemplado])
				if diferenca < taxa_mais_proxima and acidentes_dict['taxa_mun_feridos'][2015][cod_ibge] > 0:#  and variacao_taxa_mortes['PAC FIN'][cod_ibge] < 0:
					
					taxa_mais_proxima = acidentes_dict['taxa_mun_feridos'][2015][cod_ibge]
					municipio_mais_proximo = cod_ibge
					vlr_investimento_municipio_mais_proximo = 0
					try:
						vlr_investimento_municipio_mais_proximo += taxa_vlr_investimento_acum['EMENDAS'][cod_ibge]
					except KeyError:
						pass
					try:
						vlr_investimento_municipio_mais_proximo += taxa_vlr_investimento_acum['PAC FIN'][cod_ibge]
					except KeyError:
						pass
					pop_municipio_mais_proximo = populacao_dict[2015][cod_ibge]
					vlr_investimento_convertido = vlr_investimento_municipio_mais_proximo# * populacao
				#end
			#end
			# Temos o município mais próximo
			
			total_vlr_investimento += vlr_investimento_convertido
			#print("Municipio mais próximo:", municipio_mais_proximo)
			#print("Taxa do município mais próximo em 2015:", taxa_mais_proxima)
			#print("Taxa do município mais próximo em 2019:", acidentes_dict['taxa_mun_feridos'][2019][municipio_mais_proximo])
			pop_municipio_mais_proximo = populacao_dict[2015][municipio_mais_proximo]
			#print("População do município mais próximo em 2015:", pop_municipio_mais_proximo)
			variacao_mais_proximo = variacao_taxa_feridos['PAC FIN'][municipio_mais_proximo]
			#print("Variação do município",municipio_mais_proximo,":",variacao_taxa_feridos['PAC FIN'][municipio_mais_proximo])
			
			
			
			
			populacao = populacao_dict[2015][cod_ibge_nao_contemplado]
			#print("População do município não contemplado:",populacao)
			#print("Taxa de acidentes do não contemplado em 2015:", acidentes_nao_contemplados['feridos'][cod_ibge_nao_contemplado])
			#print("Taxa de acidentes do não contemplado em 2019:", acidentes_dict['taxa_mun_feridos'][2019][cod_ibge_nao_contemplado])
			variacao_nao_contemplado = acidentes_dict['taxa_mun_feridos'][2019][cod_ibge_nao_contemplado] - acidentes_dict['taxa_mun_feridos'][2015][cod_ibge_nao_contemplado]
			#print("Variação do município não contemplado:",variacao_nao_contemplado)
			
			# Total do não contemplado em 2019
			total_2019 = (acidentes_dict['taxa_mun_feridos'][2019][cod_ibge_nao_contemplado] * populacao) / 100000
			#print("\nTotal do não contemplado em 2019:", total_2019)
			
			### TÁ TUDO ERRADO AQUI NOS INTERMEDIÁRIOS, REFAZER (NO FIM DEU CERTO, MAS TEM QUE REPENSAR TUDO AQUI)
			
			
			# Usando a projeção
			taxa_2019_projecao = acidentes_dict['taxa_mun_feridos'][2019][cod_ibge_nao_contemplado] + variacao_taxa_feridos['PAC FIN'][municipio_mais_proximo]
			#print("Taxa projetada em 2019:", acidentes_dict['taxa_mun_feridos'][2019][cod_ibge_nao_contemplado], "+", variacao_taxa_feridos['PAC FIN'][municipio_mais_proximo], "=", taxa_2019_projecao)
			total_2019_projecao = (taxa_2019_projecao * populacao) / 100000
			#print("Total projetado em 2019:", total_2019_projecao)
			
			total_salvo_projecao_2019 = total_2019 - total_2019_projecao
			#print("Total salvo em 2019 (projeção):", total_salvo_projecao_2019)
			
			total_vidas_salvas += total_salvo_projecao_2019
			
			
		#end
		
	print("====>Total de feridos evitados (projeção):", total_vidas_salvas, "em", num_municipios, "municípios (população total =",total_populacao,")")
	print("(a um valor de",total_vlr_investimento/num_municipios,"por pessoa nesses municípios)")

		

#end
