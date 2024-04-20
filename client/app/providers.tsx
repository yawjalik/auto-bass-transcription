'use client'

import { ChakraProvider, extendTheme } from '@chakra-ui/react'
import { ReactNode } from 'react'

const theme = extendTheme({
  fonts: {
    heading: `'Poppins', sans-serif`,
    body: `'Poppins', sans-serif`,
    mono: `'JetBrains Mono', monospace`
  },
  colors: {
    primary: '#d809ca',
    primaryDarker: '#9d2995',
    primaryLighter: '#ff40ee',
    black: '#222222'
  },
  components: {
    Button: {
      variants: {
        primary: {
          bg: 'primary',
          color: 'white',
          _hover: {
            bg: 'primaryDarker'
          },
        },
        ghost: {
          bg: 'transparent',
          color: 'white',
          _hover: {
            bg: 'primary'
          }
        }
      }
    },
    Tabs: {
      variants: {
        line: {
          tab: {
            color: 'white',
            fontWeight: 'bold',
            _selected: {
              color: 'primaryLighter',
              borderBottomColor: 'primary'
            },
          }
        },
        enclosed: {
          tab: {
            color: 'white',
            fontWeight: 'bold',
            _selected: {
              color: 'primary',
              bg: 'primary',
              borderBottomColor: 'primary'
            }
          },
        }
      }
    },
    Switch: {
      baseStyle: {
        track: {
          bg: 'gray',
          _checked: {
            bg: 'primaryDarker'
          }
        },
      }
    },
  },
  styles: {
    global: () => ({
      'html, body': {
        fontFamily: 'body',
        color: 'white',
        backgroundColor: 'black'
      }
    })
  }
})

export function Providers({ children }: { children: ReactNode }) {
  return <ChakraProvider theme={theme}>{children}</ChakraProvider>
}
