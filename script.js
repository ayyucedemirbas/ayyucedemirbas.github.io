<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Ayyuce Demirbas</title>
  <!-- Favicon -->
  <link rel="icon" type="image/png" href="https://github.com/ayyucedemirbas/ayyucedemirbas/blob/master/logo/logo.png" />

  <!-- Styles -->
  <style>
    /* -------------------------------
       VARIABLES
    --------------------------------*/
    :root {
      --bg-light: #fefefe;
      --text-light: #111111;
      --accent-light: #0077ff;

      --bg-dark: #121212;
      --text-dark: #f0f0f0;
      --accent-dark: #bb86fc;

      /* default to dark theme */
      --bg: var(--bg-dark);
      --text: var(--text-dark);
      --accent: var(--accent-dark);

      --transition-speed: 0.3s;
      --font-stack: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* When user sets data-theme to light */
    [data-theme="light"] {
      --bg: var(--bg-light);
      --text: var(--text-light);
      --accent: var(--accent-light);
    }

    /* Basic reset */
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    /* -------------------------------
       GLOBAL STYLES
    --------------------------------*/
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-stack);
      line-height: 1.6;
      transition: background-color var(--transition-speed), color var(--transition-speed);
      margin: 0;
      padding: 0;
    }

    a {
      color: var(--accent);
      text-decoration: none;
      transition: color 0.2s;
    }
    a:hover {
      opacity: 0.8;
    }

    /* Container for layout spacing */
    .container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 1rem;
    }

    /* -------------------------------
       NAVBAR
    --------------------------------*/
    nav {
      background-color: rgba(0, 0, 0, 0.3);
      position: sticky;
      top: 0;
      width: 100%;
      z-index: 10;
      backdrop-filter: blur(10px);
    }

    nav ul {
      list-style: none;
      display: flex;
      align-items: center;
      gap: 1.5rem;
      padding: 1rem 2rem;
      margin: 0;
      font-weight: 500;
    }

    nav li {
      position: relative;
    }

    nav a {
      color: var(--text);
      font-size: 1rem;
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.05rem;
    }

    /* Example submenu styling */
    .submenu {
      cursor: pointer;
    }
    .submenu ul {
      position: absolute;
      top: 100%;
      left: 0;
      background-color: var(--bg);
      padding: 0.5rem 1rem;
      box-shadow: 0 4px 10px rgba(0,0,0,0.3);
      display: none;
      flex-direction: column;
      gap: 0.5rem;
      min-width: 150px;
      border-radius: 6px;
    }
    .submenu ul li a {
      text-transform: none;
    }
    .submenu:hover ul {
      display: flex;
    }

    /* -------------------------------
       HERO SECTION
    --------------------------------*/
    .hero {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-start;
      justify-content: space-between;
      gap: 2rem;
      margin-top: 2rem;
    }

    .hero-text {
      flex: 1 1 500px;
    }

    .hero-text h1 {
      font-size: 2.4rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
    }
    .hero-text h2 {
      font-size: 1.2rem;
      font-weight: 400;
      margin-bottom: 1rem;
      color: var(--accent);
    }
    .hero-text p {
      margin-bottom: 1rem;
      line-height: 1.8;
    }

    .awards {
      margin-top: 1rem;
    }
    .awards h3 {
      margin-bottom: 0.5rem;
    }
    .awards ul {
      list-style: disc inside;
      padding-left: 1rem;
      margin-bottom: 1rem;
    }

    /* Right-side image */
    .hero-img {
      flex: 1 1 300px;
      display: flex;
      justify-content: center;
      align-items: flex-start;
    }
    .hero-img img {
      border-radius: 8px;
      max-width: 300px;
      width: 100%;
      object-fit: cover;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }

    /* Social icons at bottom of hero text */
    .social-links {
      margin-top: 1.5rem;
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .social-links a img {
      width: 40px;
      height: 40px;
      transition: transform 0.2s;
      border-radius: 50%;
    }
    .social-links a img:hover {
      transform: scale(1.1);
    }

    /* -------------------------------
       CONTENT SECTION
    --------------------------------*/
    #content-area {
      margin: 2rem 0;
      padding: 1rem;
    }

    /* -------------------------------
       THEME SWITCHER
    --------------------------------*/
    .theme-switcher {
      position: fixed;
      bottom: 1rem;
      right: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      background-color: var(--bg);
      border: 1px solid var(--accent);
      border-radius: 6px;
      padding: 0.5rem;
    }
    .theme-switcher button {
      background: none;
      color: var(--accent);
      border: 1px solid var(--accent);
      padding: 0.3rem 0.6rem;
      border-radius: 4px;
      cursor: pointer;
      font-size: 0.8rem;
      transition: background-color 0.2s, color 0.2s;
    }
    .theme-switcher button:hover {
      background-color: var(--accent);
      color: var(--bg);
    }

    /* -------------------------------
       FOOTER
    --------------------------------*/
    footer {
      text-align: center;
      padding: 1rem 0;
      margin-top: 2rem;
      border-top: 1px solid rgba(255,255,255,0.1);
      font-size: 0.85rem;
      color: #888;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
      .hero {
        flex-direction: column;
      }
      .hero-img {
        order: -1;
        justify-content: center;
      }
    }
  </style>
</head>
<body>
  <!-- NAVBAR -->
  <nav>
    <ul>
      <li><a href="#" data-page="about">about</a></li>
      <li><a href="#" data-page="blog">blog</a></li>
      <li><a href="#" data-page="publications">publications</a></li>
      <li><a href="#" data-page="projects">projects</a></li>
      <li><a href="#" data-page="repositories">repositories</a></li>
      <li><a href="#" data-page="cv">cv</a></li>
      <li><a href="#" data-page="teaching">teaching</a></li>
      <li><a href="#" data-page="people">people</a></li>
      <li class="submenu">
        <a class="submenu-title">submenus ▼</a>
        <ul>
          <li><a href="#">Submenu 1</a></li>
          <li><a href="#">Submenu 2</a></li>
          <li><a href="#">Submenu 3</a></li>
        </ul>
      </li>
    </ul>
  </nav>

  <!-- MAIN CONTAINER -->
  <div class="container">
    <!-- HERO SECTION -->
    <section class="hero" id="hero-section">
      <div class="hero-text">
        <h1>Ayyuce Demirbas</h1>
        <h2>Machine Learning Engineer & Researcher</h2>
        <p>
          My name is Ayyuce, and I'm a machine learning engineer and a published researcher with a strong focus on vision transformers and medical images. Currently, my activity is on leveraging vision transformers for medical image analysis, aiming to contribute to the field by solving real-world problems. Additionally, I'm also interested in developing efficient large language models for natural language understanding. I hold a bachelor's degree in computer engineering and have a deep passion for machine learning. I'm looking for Ph.D. positions this fall.
        </p>
        <p>
          I’m particularly interested in staying up-to-date with the latest research and techniques in the field. As a dedicated learner, I’m always looking to seek out new challenges and explore innovative approaches. <strong>Resume is available upon request.</strong>
        </p>

        <!-- Awards section (example) -->
        <div class="awards">
          <h3>Awards:</h3>
          <ul>
            <li>Google Developer Expert in Machine Learning</li>
            <li>Microsoft Turkey Women Leaders of Technology</li>
            <li>Kaggle Grandmaster (fictional example)</li>
            <li>First place in "Medical Imaging Kaggle Challenge"</li>
          </ul>
        </div>

        <!-- Social Links -->
        <div class="social-links">
          <!-- E-mail -->
          <a href="mailto: a.ayyuced@gmail.com" target="_blank">
            <img src="https://github.com/ayyucedemirbas/ayyucedemirbas/blob/master/logo/gmail.jpg" alt="Gmail"/>
          </a>
          <!-- GitHub -->
          <a href="https://github.com/ayyucedemirbas" target="_blank">
            <img src="https://github.com/ayyucedemirbas/ayyucedemirbas/blob/master/logo/github.jpg" alt="GitHub"/>
          </a>
          <!-- Google Scholar -->
          <a href="https://scholar.google.com/citations?user=J1Zh37QAAAAJ&hl=en" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/c7/Google_Scholar_logo.svg" alt="Google Scholar"/>
          </a>
          <!-- ORCID -->
          <a href="https://orcid.org/0000-0002-6731-9345" target="_blank">
            <img src="https://cdn.simpleicons.org/orcid" alt="ORCID"/>
          </a>
          <!-- ResearchGate -->
          <a href="https://www.researchgate.net/profile/Ayse-Demirbas" target="_blank">
            <img src="https://cdn.simpleicons.org/researchgate" alt="ResearchGate"/>
          </a>
          <!-- Hugging Face -->
          <a href="https://huggingface.co/ayyuce" target="_blank">
            <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face"/>
          </a>
          <!-- Kaggle -->
          <a href="https://www.kaggle.com/ayyuce" target="_blank">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" alt="Kaggle"/>
          </a>
          <!-- GitLab -->
          <a href="https://gitlab.com/ayyuce" target="_blank">
            <img src="https://github.com/ayyucedemirbas/ayyucedemirbas/blob/master/logo/gitlab.png" alt="GitLab"/>
          </a>
          <!-- Keybase -->
          <a href="https://keybase.io/ayyuce" target="_blank">
            <img src="https://github.com/ayyucedemirbas/ayyucedemirbas/blob/master/logo/keybase.png" alt="Keybase"/>
          </a>
          <!-- Medium -->
          <a href="https://medium.com/@ayyucedemirbas" target="_blank">
            <img src="https://github.com/ayyucedemirbas/ayyucedemirbas/blob/master/logo/medium.jpeg" alt="Medium"/>
          </a>
          <!-- Goodreads -->
          <a href="https://www.goodreads.com/ayyuce" target="_blank">
            <img src="https://github.com/ayyucedemirbas/ayyucedemirbas/blob/master/logo/goodreads.png" alt="Goodreads"/>
          </a>
          <!-- X -->
          <a href="https://x.com/demirbasayyuce" target="_blank">
            <img src="https://github.com/ayyucedemirbas/ayyucedemirbas/blob/master/logo/x.png" alt="X"/>
          </a>
        </div>
      </div>

      <!-- Hero Image -->
      <div class="hero-img">
        <img 
          src="https://github.com/ayyucedemirbas/ayyucedemirbas/blob/master/logo/IMG_0126.JPG"
          alt="Ayyuce's Photo"
        />
      </div>
    </section>

    <!-- MARKDOWN CONTENT AREA -->
    <section id="content-area">
      <!-- When you click on nav links (other than about), 
           we load the corresponding .md file here. -->
    </section>
  </div>

  <!-- THEME SWITCHER (Optional) -->
  <div class="theme-switcher" id="themeSwitcher">
    <button onclick="switchTheme('light')">Light</button>
    <button onclick="switchTheme('dark')">Dark</button>
    <button onclick="switchTheme('system')">System</button>
  </div>

  <!-- FOOTER -->
  <footer>
    <p>&copy; 2025 Ayyuce Demirbas. All Rights Reserved.</p>
  </footer>

  <!-- MARKED.JS for Markdown parsing -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script>
    // ----------------------------
    // THEME SWITCHING
    // ----------------------------
    function switchTheme(theme) {
      if (theme === 'system') {
        document.documentElement.removeAttribute('data-theme');
      } else {
        document.documentElement.setAttribute('data-theme', theme);
      }
      localStorage.setItem('preferred-theme', theme);
    }

    (function initTheme() {
      const savedTheme = localStorage.getItem('preferred-theme') || 'dark';
      if (savedTheme === 'system') {
        document.documentElement.removeAttribute('data-theme');
      } else {
        document.documentElement.setAttribute('data-theme', savedTheme);
      }
    })();

    // ----------------------------
    // NAV & MARKDOWN LOADER
    // ----------------------------
    const navLinks = document.querySelectorAll('nav a[data-page]');
    const contentArea = document.getElementById('content-area');

    navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const page = link.getAttribute('data-page');
        // "about" just scrolls back up to hero
        if (page === 'about') {
          document.getElementById('hero-section').scrollIntoView({ behavior: 'smooth' });
          contentArea.innerHTML = '';
        } else {
          // load the .md file from "content/<page>.md"
          fetch(`content/${page}.md`)
            .then(resp => {
              if (!resp.ok) throw new Error('Not found');
              return resp.text();
            })
            .then(md => {
              contentArea.innerHTML = marked.parse(md);
              contentArea.scrollIntoView({ behavior: 'smooth' });
            })
            .catch(err => {
              contentArea.innerHTML = `<p style="color: var(--accent)">Content not found for <strong>${page}</strong>.</p>`;
            });
        }
      });
    });
  </script>
</body>
</html>
