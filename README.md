
# 🧠 BioMovie – Identificação Biométrica em Vídeos

[🔗 Live Demo](https://projetos.tiago.cafe/deep/)  
[📦 Repositório no GitHub](https://github.com/algodas/biomovie)

O **BioMovie** é uma aplicação Flask que realiza identificação facial automática em vídeos a partir de uma imagem de referência. Com uma arquitetura leve e altamente funcional, o sistema compara rostos extraídos de um vídeo com a imagem enviada pelo usuário, verifica similaridade e identifica se a pessoa possui uma **notação registrada previamente** (ex: alerta de desaparecido, procurado, etc).

---

### 📌 Funcionalidades

- 🧬 **Reconhecimento facial 1:N**: compara a imagem de referência com múltiplas faces detectadas no vídeo.
- 🎯 **Detecção por frame**: varredura com saltos entre frames para eficiência no processamento.
- ⚠️ **Sistema de notificação**: alerta se a pessoa identificada possui imagem cadastrada na pasta de observações (`notificacao/`).
- 📊 **Pontuação de similaridade**: exibe a distância vetorial e o percentual de confiança entre rostos.
- 📂 **Upload de arquivos** com feedback visual.
- 🧪 Baseado no modelo **ArcFace** via [DeepFace](https://github.com/serengil/deepface).

---

### 📂 Estrutura do Projeto

```
biomovie/
├── app6.py                   # Servidor Flask e lógica principal
├── templates/
│   └── index.html            # Página web principal com formulário
├── static/
│   └── style.css             # Estilos complementares (opcional)
├── models/                   # Modelos DNN de detecção facial (OpenCV)
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── Uploads/videos/           # Vídeos recebidos
└── temp/                     # Imagens temporárias para processamento
```

---

### 🚀 Como Usar

1️⃣ Acesse a [Live Demo](https://projetos.tiago.cafe/deep/)  
2️⃣ Faça o upload da **foto de referência**.  
3️⃣ Faça o upload de um **vídeo** no formato `.mp4`.  
4️⃣ Clique em **Comparar Faces**.  

🧾 O sistema irá retornar:

- As **pessoas detectadas** no vídeo.
- O **frame** onde cada pessoa foi localizada.
- A **pontuação de similaridade** com a imagem enviada.
- ⚠️ Alerta visual caso alguma das pessoas identificadas no vídeo esteja cadastrada previamente com alguma restrição.

🔍 Exemplo de arquivos de entrada: [📥 Download](https://projetos.tiago.cafe/demo_images/videos.zip)

---

### 🧠 Tecnologias Utilizadas

- `Python` / `Flask`  
- `OpenCV` (extração de frames)  
- `DeepFace` (comparação com ArcFace)  
- `PIL`, `NumPy`, `Werkzeug`, `dotenv`

---

### ✅ Requisitos

- Python 3.8+
- Instale as dependências:

```bash
pip install -r requirements.txt
```

---

### 🧪 Exemplo de Casos

Nos vídeos de exemplo disponíveis para download, **três pessoas possuem observações previamente cadastradas** e serão destacadas quando reconhecidas.

---

### 🛡️ Observações Legais

Os vídeos utilizados são de domínio público, extraídos do YouTube sob licença de reprodução curta. São utilizados apenas para fins educacionais e não têm qualquer relação real com pessoas com registros oficiais.
