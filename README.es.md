# Vault de Obsidian sobre IA / ML

Versión en español. English version: [README.md](./README.md)

Este repositorio es un vault de Obsidian con notas sobre inteligencia artificial, machine learning, NLP y sistemas agénticos.

Está diseñado para funcionar en dos modos:
- **Modo referencia**: entrar en una nota concreta y entender el concepto rápido.
- **Modo estudio**: empezar por la capa de índices y seguir rutas de aprendizaje guiadas.

## Inicio Rápido
1. Clona o descarga este repositorio.
2. Abre la carpeta en Obsidian como un vault existente.
3. Empieza en [`00 Home/Home.md`](./00%20Home/Home.md).

## Configuración Inicial Si Nunca Has Usado Obsidian
Si es tu primera vez con Obsidian, sigue esta secuencia:

1. Instala Obsidian desde [obsidian.md](https://obsidian.md/).
2. Abre Obsidian.
3. En la pantalla inicial, elige **Open folder as vault** o **Open existing vault**.
4. Selecciona la carpeta descargada del repositorio.
5. Cuando se abra el vault, empieza en [`00 Home/Home.md`](./00%20Home/Home.md).

## Instalación de Plugins Para Usuarios Nuevos
El vault se puede leer sin plugins extra, pero la experiencia completa depende de un plugin core y un plugin community.

### 1. Activar Bases
`Bases` es un plugin core de Obsidian.

1. Abre **Settings**.
2. Ve a **Core plugins**.
3. Busca `Bases`.
4. Actívalo.

Lo necesitas para:
- [`00 Home/Vault Catalog.base`](./00%20Home/Vault%20Catalog.base)
- [`90 Guides/Editorial Review.base`](./90%20Guides/Editorial%20Review.base)

### 2. Activar los Community Plugins
`Dataview` es un community plugin. Por defecto, Obsidian mantiene los community plugins desactivados en Restricted Mode.

1. Abre **Settings**.
2. Ve a **Community plugins**.
3. Lee el aviso y continúa solo si confías en este vault y en su configuración.
4. Pulsa **Turn on community plugins** o desactiva **Restricted mode**.

Referencia oficial de seguridad:
- [Obsidian Plugin Security](https://obsidian.md/help/plugin-security)

### 3. Activar Dataview
Este repositorio ya incluye los archivos del plugin Dataview en `.obsidian/plugins/dataview`, pero Obsidian igualmente necesita que el plugin quede activado en tu vault local.

Si Dataview no aparece activo automáticamente:
1. Abre **Settings**.
2. Ve a **Community plugins**.
3. Busca `Dataview` en la lista de plugins instalados.
4. Actívalo.

Si no aparece listado por cualquier motivo:
1. Abre **Settings**.
2. Ve a **Community plugins**.
3. Pulsa **Browse**.
4. Busca `Dataview`.
5. Instálalo.
6. Actívalo.

Lo necesitas para:
- [`00 Home/Vault Dashboard.md`](./00%20Home/Vault%20Dashboard.md)
- [`90 Guides/Editorial Dashboard.md`](./90%20Guides/Editorial%20Dashboard.md)

## Configuración Recomendada De Obsidian
Para la experiencia completa, usa una versión reciente de Obsidian y mantén la carpeta `.obsidian` incluida en el repositorio.

### Obligatorio Para La Funcionalidad Completa
- **Dataview** community plugin: necesario para los dashboards en:
  - [`00 Home/Vault Dashboard.md`](./00%20Home/Vault%20Dashboard.md)
  - [`90 Guides/Editorial Dashboard.md`](./90%20Guides/Editorial%20Dashboard.md)
- **Bases** core plugin: usado por:
  - [`00 Home/Vault Catalog.base`](./00%20Home/Vault%20Catalog.base)
  - [`90 Guides/Editorial Review.base`](./90%20Guides/Editorial%20Review.base)

### Opcional Pero Recomendado
- Mantener el `workspace` incluido si quieres la apertura guiada del vault.
- Mantener el snippet de Mermaid si quieres que los diagramas no desborden el ancho del panel.

## Qué Pasa Si No Activas Los Plugins
- Sin **Bases**, los archivos `.base` no serán útiles.
- Sin **Dataview**, los dashboards se abrirán, pero sus bloques de consulta no se renderizarán correctamente.
- Sin ambos, el vault sigue siendo perfectamente usable como base de conocimiento en Markdown a través de la portada y de los índices.

## Qué Funciona Incluso Sin Plugins Extra
Estas partes funcionan como notas Markdown normales aunque no tengas Dataview:
- estructura de carpetas
- wiki-links
- contenido de las notas
- frontmatter
- índices principales

Si Dataview no está activado, el vault sigue siendo usable, pero los dashboards pierden parte de su valor.

## Por Dónde Empezar
- [`00 Home/Home.md`](./00%20Home/Home.md): portal principal
- [`00 Home/Machine Learning Index.md`](./00%20Home/Machine%20Learning%20Index.md): ruta general de ML
- [`00 Home/Data Preparation Index.md`](./00%20Home/Data%20Preparation%20Index.md): rama de preprocesado
- [`00 Home/Deep Learning & NLP Index.md`](./00%20Home/Deep%20Learning%20%26%20NLP%20Index.md): deep learning, NLP, LLMs y RAG
- [`00 Home/Agentic Systems Index.md`](./00%20Home/Agentic%20Systems%20Index.md): agentes, orquestación y sistemas multiagente

## Estructura Del Vault
- `00 Home`: portal principal, índices, dashboards y vistas de Bases
- `01 Foundations`: estadística, sesgo y conceptos base de datos
- `02 Data Preparation`: encoding, imputación, escalado y datasets desbalanceados
- `03 Classical ML`: métricas, modelos lineales, árboles y ML tabular
- `04 Deep Learning & NLP`: redes neuronales, secuencias, NLP, language models y RAG
- `05 Agentic Systems`: agentes, tool use, planificación, memoria, orquestación y evaluación
- `90 Guides`: guía de estilo y dashboard editorial
- `99 Archive`: reservado para notas obsoletas

## Notas Para Colaboradores
Si quieres extender o mantener el vault, usa:
- [`AGENTS.md`](./AGENTS.md): reglas operativas cortas para agentes de IA
- [`90 Guides/Note Style Guide.md`](./90%20Guides/Note%20Style%20Guide.md): guía canónica de autoría, curriculum, metadata y dashboards

## Notas Sobre Portabilidad
- Cualquier usuario de Obsidian puede abrir el vault directamente.
- El archivo `.obsidian/workspace.json` es opinado; si alguien prefiere otro layout local, puede cambiarlo sin afectar el contenido.
- Parte del comportamiento visual depende del tema, de los plugins y de la configuración local de Obsidian.
- Los nombres exactos de algunos menús pueden variar ligeramente entre versiones, pero el flujo general es el mismo: abrir el vault, activar `Bases` y después activar `Dataview`.
