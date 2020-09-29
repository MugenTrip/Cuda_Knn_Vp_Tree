// C program for array implementation of stack 
#include <limits.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <stdbool.h>
#include "vptree.h"

// A structure to represent a stack 
typedef struct Stack { 
	int top; 
	unsigned capacity; 
	vptree *array; 
}Stack; 

// function to create a stack of given capacity. It initializes size of 
// stack as 0 
struct Stack* createStack(unsigned capacity , int d) 
{ 
	struct Stack* stack = (struct Stack*)malloc(sizeof(struct Stack)); 
	stack->capacity = capacity; 
	stack->top = -1; 
	stack->array = (vptree*)malloc(stack->capacity * sizeof(vptree));
	for (int i = 0; i < stack->capacity; i++)
	{
		stack->array[i].vantage_point = (double *) malloc(sizeof(double)*d);
	}
	return stack; 
} 

// Stack is full when top is equal to the last index 
int isFull(struct Stack* stack) 
{ 
	return stack->top == stack->capacity - 1; 
} 

// Stack is empty when top is equal to -1 
int isEmpty(struct Stack* stack) 
{ 
	return stack->top == -1; 
} 

// Function to add an item to stack. It increases top by 1 
void push(struct Stack* stack, vptree *node,int d) 
{ 
	++stack->top;
	if (isFull(stack)) 
		return; 
	stack->array[stack->top].idx = node->idx;
	stack->array[stack->top].median_value = node->median_value;
	for (int i = 0; i < d; i++)
	{
	 	stack->array[stack->top].vantage_point[i] = node->vantage_point[i];
	}
	stack->array[stack->top].inner = node->inner;
	stack->array[stack->top].outer = node->outer;

} 

// Function to remove an item from stack. It decreases top by 1 
int pop(struct Stack* stack , vptree *node, int d) 
{ 
	if (isEmpty(stack)) 
		return 0;
	node->idx = stack->array[stack->top].idx;
	node->median_value = stack->array[stack->top].median_value;
	for (int i = 0; i < d; i++)
	{
	 	node->vantage_point[i] = stack->array[stack->top].vantage_point[i];
	}
	node->inner = stack->array[stack->top].inner;
	node->outer = stack->array[stack->top].outer;
	stack->top--;
	return 1; 
} 

// Function to return the top from stack without removing it 
vptree* peek(struct Stack* stack) 
{ 
	if (isEmpty(stack)) 
		return NULL; 
	return &stack->array[stack->top]; 
} 