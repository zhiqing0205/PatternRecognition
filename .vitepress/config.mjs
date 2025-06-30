import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({
  outDir: process.env.VITE_BUILD_OUTDIR || '.vitepress/dist',
  title: '模式识别',
  description: '模式识别课程学习笔记',
  base: '/', 
  
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '章节目录', link: '/docs/chapters/chapter01' },
      { text: '数学基础', link: '/docs/chapters/appendix-math' }
    ],

    sidebar: {
      '/docs/chapters/': [
        {
          text: '章节目录',
          items: [
            { text: '第1章 绪论', link: '/docs/chapters/chapter01' },
            { text: '第2章 贝叶斯决策论', link: '/docs/chapters/chapter02' },
            { text: '第3章 最大似然估计和贝叶斯参数估计(重点)', link: '/docs/chapters/chapter03' },
            { text: '第4章 非参数技术', link: '/docs/chapters/chapter04' },
            { text: '第5章 线性判别函数(重点)', link: '/docs/chapters/chapter05' },
            { text: '第8章 非度量方法', link: '/docs/chapters/chapter08' },
            { text: '第9章 独立于算法的机器学习(重点)', link: '/docs/chapters/chapter09' },
            { text: '第10章 无监督学习与聚类', link: '/docs/chapters/chapter10' }
          ]
        },
        {
          text: '附录',
          items: [
            { text: '数学基础', link: '/docs/chapters/appendix-math' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/zhiqing0205/PatternRecognition' }
    ],

    footer: {
      message: '模式识别课程学习笔记',
      copyright: 'Copyright © 2025'
    },

    search: {
      provider: 'local'
    },

    outline: {
      level: [2, 3],
      label: '页面导航'
    }
  },

  markdown: {
    lineNumbers: true,
    config: (md) => {
      md.use(mathjax3)
    }
  },

  head: [
    [
      'script',
      { id: 'MathJax-script', async: true, src: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js' }
    ],
    [
      'script',
      {},
      `
      window.MathJax = {
        tex: {
          inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
          processEscapes: true,
          processEnvironments: true
        },
        options: {
          skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
        },
        svg: {
          scale: 1,
          displayAlign: 'center',
          displayIndent: '0',
          fontCache: 'local',
          internalSpeechTitles: true
        },
        startup: {
          ready: () => {
            MathJax.startup.defaultReady();
            // 移除所有滚动条样式
            const style = document.createElement('style');
            style.textContent = \`
              mjx-container[display="true"] {
                overflow: visible !important;
                width: auto !important;
                max-width: none !important;
              }
              mjx-container[display="true"] svg {
                overflow: visible !important;
                width: auto !important;
                max-width: none !important;
              }
            \`;
            document.head.appendChild(style);
          }
        }
      }
      `
    ]
  ]
})