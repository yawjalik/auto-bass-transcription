import { Text, VStack } from '@chakra-ui/react'
import { ReactNode } from 'react'

const CardButton = ({
  text,
  icon,
  ...rest
}: {
  text: string
  icon: ReactNode
  [_: string]: any
}) => {
  return (
    <VStack
      border="1px solid lightgray"
      borderRadius="md"
      p={4}
      _hover={{
        bg: '#e9e9e9',
        color: 'black',
        borderColor: '#e9e9e9',
        cursor: 'pointer',
        transition: 'all 0.3s'
      }}
      {...rest}
    >
      {icon}
      <Text fontWeight="bold">{text}</Text>
    </VStack>
  )
}

export default CardButton
