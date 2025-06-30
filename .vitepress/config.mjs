import { defineConfig } from 'vitepress'

export default defineConfig({
  title: '模式识别',
  description: '模式识别课程学习笔记',
  base: '/',
  
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '章节目录', link: '/docs/chapters/chapter01' }
    ],

    sidebar: {
      '/docs/chapters/': [
        {
          text: '课程内容',
          items: [
            { text: '第1章 绪论', link: '/docs/chapters/chapter01' },
            { text: '第2章 贝叶斯决策论', link: '/docs/chapters/chapter02' },
            { text: '第3章 最大似然估计和贝叶斯参数估计', link: '/docs/chapters/chapter03' },
            { text: '第4章 非参数技术', link: '/docs/chapters/chapter04' },
            { text: '第5章 线性判别函数', link: '/docs/chapters/chapter05' },
            { text: '第8章 非度量方法', link: '/docs/chapters/chapter08' },
            { text: '第9章 独立于算法的机器学习', link: '/docs/chapters/chapter09' },
            { text: '第10章 无监督学习与聚类', link: '/docs/chapters/chapter10' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com' }
    ],

    footer: {
      message: '模式识别课程学习笔记',
      copyright: 'Copyright © 2024'
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
    lineNumbers: true
  }
})